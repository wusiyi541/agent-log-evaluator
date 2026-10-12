import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openpyxl import load_workbook

from config import AppConfig, parse_dates
from log_parser import LogParser, LogTurn
from judge_engine import JudgeEngine, EvalResult


NEGATIVE_REFERENCE_ANSWER = "抱歉，当前政务数据共享平台知识库暂未收录该问题相关政策、操作规范、应用案例以及数据资源信息，无法为您提供准确答复，您可联系运营人员咨询核实。"


def load_env_file(env_path: Optional[str] = None):
    if env_path and os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv()


def load_config(
    log_file: Optional[str] = None,
    test_set_file: Optional[str] = None,
    mapping_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    dates: Optional[str] = None,
    keywords: Optional[str] = None,
) -> AppConfig:
    config = AppConfig()

    if log_file:
        config.log_parser.log_file = log_file
    if test_set_file:
        config.test_set.test_set_file = test_set_file
    if mapping_file:
        config.test_set.mapping_file = mapping_file
    if output_dir:
        config.output.output_dir = output_dir
    if dates:
        config.log_parser.target_dates = parse_dates(dates)
    if keywords:
        config.log_parser.target_keywords = [k.strip() for k in keywords.split(",")]

    if not config.log_parser.log_file:
        print("[ERROR] 日志文件路径未设置")
        sys.exit(1)

    if not os.path.exists(config.log_parser.log_file):
        print(f"[ERROR] 日志文件不存在: {config.log_parser.log_file}")
        sys.exit(1)

    if not os.path.exists(config.output.output_dir):
        os.makedirs(config.output.output_dir, exist_ok=True)

    return config


def load_test_cases(config: AppConfig) -> List[Dict[str, Any]]:
    test_set_file = config.test_set.test_set_file

    if not test_set_file or not os.path.exists(test_set_file):
        print("[WARN] 测试集文件未找到，将仅解析日志")
        return []

    print(f"\n[INFO] 加载测试集: {test_set_file}")

    import pandas as pd

    test_cases = []

    header_row = config.test_set.excel_header_row
    question_col = config.test_set.excel_question_col
    answer_col = config.test_set.excel_answer_col
    id_col = config.test_set.excel_id_col
    scenario_col = config.test_set.excel_scenario_col
    is_negative_col = config.test_set.excel_is_negative_col

    # 评测批跑兼容性：xlsx 可能多 sheet（README / 测试集_主数据 / 字典 …），优先选命名命中的或列数≥7的主表 sheet
    xls = pd.ExcelFile(test_set_file)
    preferred_names = {"测试集_主数据", "测试用例", "主数据", "Sheet1"}
    sheet_names = xls.sheet_names
    selected_sheet = next((s for s in sheet_names if s in preferred_names), None)
    if selected_sheet is None:
        # 无命名时挑列数最多、且列数≥id/question/answer最大列索引+1的 sheet，避免误取 README 单列表
        best_sheet = None
        best_score = -1
        needed_cols = max(id_col, question_col, answer_col, scenario_col, is_negative_col) + 1
        for s in sheet_names:
            try:
                df_probe = pd.read_excel(xls, sheet_name=s, header=header_row, nrows=5)
            except Exception:
                continue
            score = len(df_probe.columns)
            if score >= needed_cols and score > best_score:
                best_sheet = s
                best_score = score
        selected_sheet = best_sheet or (sheet_names[0] if sheet_names else 0)
    df = pd.read_excel(xls, sheet_name=selected_sheet, header=header_row)
    print(f"[INFO] 选用 sheet='{selected_sheet}'；总 {len(sheet_names)} 个 sheet")
    col_names = [str(c) for c in df.columns.tolist()]
    print(f"[INFO] Excel列名: {col_names[:20]}")
    print(f"[INFO] 总行数: {len(df)}")

    # 按列名优先匹配（兼容 v2-v6 / 旧版单 sheet 等不同版本模板）；找不到再回退到 .env 指定的列索引
    def _col(aliases, fallback_idx):
        for alias in aliases:
            for idx, cname in enumerate(col_names):
                if alias in cname and len(cname) <= 2 * len(alias) + 2:  # 避免"问题"被"标准问题核心要点"等超集误匹配
                    return idx
        return fallback_idx

    id_col = _col(["样本编号", "用例ID", "用例编号", "测试用例ID", "案例ID"], id_col)
    scenario_col = _col(["业务场景", "场景", "场景分类"], scenario_col)
    question_col = _col(["用户问题", "问题", "提问"], question_col)
    # 标准完整答案 优先级最高，再退到 标准答案/答案 类
    answer_col = _col(["标准完整答案", "标准答案完整", "完整标准答案"], -1)
    if answer_col == -1:
        answer_col = _col(["标准核心答案", "参考答案", "标准答案核心", "标准回答", "标准答案", "答案"], answer_col)
    is_negative_col = _col(["是否负面", "负样本标记", "合规标签", "是否负样本", "负样本", "风险标签"], is_negative_col)
    print(f"[INFO] 解析用列（名匹配后）: id={id_col} 场景={scenario_col} 问题={question_col} "
          f"答案={answer_col} 负样本={is_negative_col}")

    # 轻量健康检查：若问题列抽出的前 3 条都形如 YYYY-MM-DD（错把日期列当问题 → 日志告警并 skip）
    bad_question_col = False
    if len(df) >= 3:
        import re as _re
        samples = []
        for ri in range(min(3, len(df))):
            r = df.iloc[ri]
            if question_col < len(r) and pd.notna(r.iloc[question_col]):
                samples.append(str(r.iloc[question_col]))
        if samples and all(_re.match(r'^\d{4}-\d{2}-\d{2}', s) for s in samples):
            print(f"[WARN] 问题列 {question_col} 前 3 条均为日期: {samples}，判定为错列；尝试强制找'用户问题'所在列")
            bad_question_col = True
            for idx, cname in enumerate(col_names):
                if cname == "用户问题":
                    question_col = idx
                    print(f"[INFO] 已强制纠正: 问题列={question_col}")
                    break

    for row_idx, row in df.iterrows():
        def _get(idx, default=""):
            return (str(row.iloc[idx]) if idx < len(row) and pd.notna(row.iloc[idx]) else default) or default

        test_id = _get(id_col, f"T{row_idx + 1}")
        question = _get(question_col, "")
        reference_answer = _get(answer_col, "")
        scenario = _get(scenario_col, "")
        is_negative_raw = _get(is_negative_col, "")
        if is_negative_raw == "nan":
            is_negative_raw = ""
        # 多版本值归一："是"/"否" 或 不合规/合规 或 Negative/Positive
        is_negative = "是" if (is_negative_raw == "是" or "不合规" in is_negative_raw or
                               is_negative_raw.lower() in {"negative", "true", "1", "bad"}) else ""

        if is_negative == "是":
            reference_answer = NEGATIVE_REFERENCE_ANSWER

        if not question or bad_question_col and _re.match(r'^\d{4}-\d{2}-\d{2}', question):
            continue
        if question == "nan":
            continue
        if len(question.strip()) < 2:
            continue

        test_cases.append(
            {
                "id": test_id,
                "question": question,
                "reference_answer": reference_answer,
                "scenario": scenario,
                "is_negative": is_negative,
                "row_number": row_idx + 1,
            }
        )

    print(f"[INFO] 加载了 {len(test_cases)} 条测试用例")
    
    scenario_count = {}
    negative_count = 0
    for tc in test_cases:
        s = tc.get("scenario", "")
        scenario_count[s] = scenario_count.get(s, 0) + 1
        if tc.get("is_negative") == "是":
            negative_count += 1
    print(f"[INFO] 业务场景分布: {scenario_count}")
    print(f"[INFO] 负样本数量: {negative_count}")

    return test_cases


def load_mapping_file(mapping_file: str) -> Dict[str, int]:
    mapping = {}

    if not mapping_file or not os.path.exists(mapping_file):
        return mapping

    print(f"\n[INFO] 加载映射文件: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        for test_id, turn_index in data.items():
            mapping[str(test_id)] = int(turn_index)
    except json.JSONDecodeError:
        for line in content.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                test_id, turn_idx = line.split(":", 1)
                mapping[test_id.strip()] = int(turn_idx.strip())

    print(f"[INFO] 加载了 {len(mapping)} 条映射关系")

    return mapping


def match_turns_to_test_cases(
    test_cases: List[Dict[str, Any]],
    turns: List[LogTurn],
    mapping: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    matched = []
    used_indices = set()

    if mapping:
        print("\n[INFO] 使用手动映射文件匹配")
        for tc in test_cases:
            test_id = tc["id"]
            if test_id in mapping:
                turn_idx = mapping[test_id]
                if 0 <= turn_idx < len(turns):
                    tc_copy = tc.copy()
                    tc_copy["log_turn"] = turns[turn_idx]
                    matched.append(tc_copy)
                    used_indices.add(turn_idx)
                    continue
            tc_copy = tc.copy()
            tc_copy["log_turn"] = None
            matched.append(tc_copy)

        for tc in matched:
            if tc.get("log_turn") is None:
                tc["log_turn"] = auto_match_single(tc["question"], turns, used_indices)
                if tc["log_turn"]:
                    used_indices.add(turns.index(tc["log_turn"]))
    else:
        print("\n[INFO] 使用自动匹配")
        for tc in test_cases:
            tc_copy = tc.copy()
            tc_copy["log_turn"] = auto_match_single(tc["question"], turns, used_indices)
            if tc_copy["log_turn"]:
                used_indices.add(turns.index(tc_copy["log_turn"]))
            matched.append(tc_copy)

    matched_count = sum(1 for m in matched if m.get("log_turn"))
    print(f"[INFO] 匹配成功: {matched_count}/{len(test_cases)}")

    return matched


DOMAIN_KEYWORDS = {
    "身份": ["身份", "人口", "户籍", "居民"],
    "教育": ["教育", "学籍", "学历", "学位", "学生", "学校", "高考"],
    "婚姻": ["婚姻", "结婚", "离婚", "夫妻"],
    "民政": ["民政", "低保", "残疾人", "救助"],
    "社保": ["社保", "养老", "保险", "退休"],
    "医疗": ["医疗", "医院", "健康", "就诊"],
    "企业": ["企业", "公司", "工商", "注册", "经营"],
    "数据": ["数据", "信息", "目录", "资源"],
    "申请": ["申请", "办理", "流程", "步骤"],
    "查询": ["查询", "查", "检索", "获取"],
    "政务数据共享": ["政务数据共享", "共享", "数据共享"],
    "目录": ["目录", "编目", "挂接", "资源目录"],
    "应用": ["应用", "案例", "场景", "使用"],
    "办事": ["办事", "业务", "事项", "服务"],
    "政策": ["政策", "条例", "规定", "法规"],
    "指导": ["指导", "操作", "指引", "指南"],
}


def extract_keywords(text: str) -> List[str]:
    keywords = set()
    for key, synonyms in DOMAIN_KEYWORDS.items():
        if key in text:
            keywords.add(key)
        for syn in synonyms:
            if syn in text:
                keywords.add(key)
                break
    return list(keywords)


def keyword_overlap_score(text1: str, text2: str) -> float:
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    common = set(keywords1) & set(keywords2)
    union = set(keywords1) | set(keywords2)
    
    if not union:
        return 0.0
    
    return len(common) / len(union)


def auto_match_single(
    question: str,
    turns: List[LogTurn],
    used_indices: set,
) -> Optional[LogTurn]:
    if not question or not turns:
        return None

    candidates = []
    for idx, turn in enumerate(turns):
        if idx in used_indices:
            continue

        turn_question = turn.user_question
        if not turn_question:
            continue

        if question.strip() == turn_question.strip():
            return turn

        sim_score = calculate_similarity(question, turn_question)
        kw_score = keyword_overlap_score(question, turn_question)
        combined_score = sim_score * 0.7 + kw_score * 0.3

        if combined_score >= 0.5:
            candidates.append((combined_score, idx, turn))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]

    return None


def calculate_similarity(text1: str, text2: str) -> float:
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    if text1 == text2:
        return 1.0

    len1, len2 = len(text1), len(text2)
    if len1 == 0 or len2 == 0:
        return 0.0

    common_chars = sum(1 for c in text1 if c in text2)
    char_sim = common_chars / max(len1, len2)

    words1 = set(text1.split())
    words2 = set(text2.split())
    if words1 and words2:
        common_words = words1.intersection(words2)
        word_sim = len(common_words) / len(words1.union(words2))
    else:
        word_sim = 0.0

    return 0.6 * char_sim + 0.4 * word_sim


def run_evaluation(
    config: AppConfig,
    turn_range: Optional[str] = None,
    test_range: Optional[str] = None,
):
    print("=" * 60)
    print("日志评测系统")
    print("=" * 60)

    # 传递测试集文件给 LogParser，同时透传时间窗/轮次策略等新参数（审查 #12 env 契约）
    test_set_file = config.test_set.test_set_file if config.test_set.test_set_file else None
    parser = LogParser(
        config.log_parser.log_file,
        config.log_parser.target_dates,
        config.log_parser.target_keywords,
        target_questions=getattr(config.log_parser, "target_questions", None),
        test_set_file=test_set_file,
        target_time_after=config.log_parser.target_time_after,
        target_time_before=getattr(config.log_parser, "target_time_before", None),
        turn_boundary=getattr(config.log_parser, "turn_boundary", "intent_anchor"),
        parse_catalog_metadata=getattr(config.log_parser, "parse_catalog_metadata", True),
    )

    turns = parser.parse()

    if not turns:
        print("[ERROR] 未解析到任何日志记录")
        return

    stats = parser.get_statistics()
    print("\n" + "=" * 60)
    print("日志统计")
    print("=" * 60)
    print(f"总对话轮次: {stats['total_turns']}")
    print(f"总耗时: {stats['total_cost_seconds']}秒")
    print(f"平均耗时: {stats['avg_cost_ms']}ms")
    print(f"最大耗时: {stats['max_cost_ms']}ms")
    print(f"总API调用: {stats['total_api_calls']}次")
    print(f"平均API调用: {stats['avg_api_calls']}次/轮")
    if stats.get("tool_distribution"):
        print("工具分布:")
        for tool, count in stats["tool_distribution"].items():
            print(f"  - {tool}: {count}次")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_output_file = os.path.join(config.output.output_dir, f"parsed_logs_{timestamp}.json")
    parser.export_to_json(log_output_file)

    if turn_range:
        start, end = parse_range(turn_range, len(turns))
        turns = turns[start:end]
        print(f"\n[INFO] 选中日志范围: [{start}:{end}], 共 {len(turns)} 条")

    test_cases = load_test_cases(config)

    if not test_cases:
        print("\n[INFO] 无测试集，仅解析日志")
        return {
            "turns": [t.to_dict() for t in turns],
            "stats": stats,
            "timestamp": timestamp,
        }

    if test_range:
        start, end = parse_range(test_range, len(test_cases))
        test_cases = test_cases[start:end]
        print(f"\n[INFO] 选中测试范围: [{start}:{end}], 共 {len(test_cases)} 条")

    mapping = {}
    if config.test_set.mapping_file:
        mapping = load_mapping_file(config.test_set.mapping_file)

    matched = match_turns_to_test_cases(test_cases, turns, mapping)

    engine = JudgeEngine(config.judge, config.eval)

    results = engine.evaluate_batch(matched)

    summary = engine.get_summary(results)

    print("\n" + "=" * 60)
    print("评测总结")
    print("=" * 60)
    print(f"有效评测数: {summary.get('valid_count', 0)}")
    print(f"通过率: {summary.get('pass_rate', 0)}%")
    print(f"综合平均分: {summary.get('overall_avg', 0)}")

    dim_names = {
        "accuracy": "准确性",
        "completeness": "完整性",
        "safety": "安全性",
        "readability": "可读性",
        "latency": "响应时长",
    }

    print("\n各维度详情:")
    for dim_key, dim_info in summary.get("dimensions", {}).items():
        print(f"  {dim_names.get(dim_key, dim_key)}:")
        print(f"    平均分: {dim_info.get('avg_score', 0)}")
        print(f"    >=4分占比: {dim_info.get('over_4_rate', 0)}%")

    output_data = {
        "timestamp": timestamp,
        "config": {
            "log_file": config.log_parser.log_file,
            "test_set_file": config.test_set.test_set_file,
            "dates": list(config.log_parser.target_dates) if config.log_parser.target_dates else None,
            "keywords": config.log_parser.target_keywords,
            "judge_model": config.judge.model,
            "batch_time_after": config.log_parser.target_time_after,
            "batch_time_before": getattr(config.log_parser, "target_time_before", None),
        },
        "log_stats": stats,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }

    eval_output_file = os.path.join(
        config.output.output_dir, f"eval_report_{timestamp}.json"
    )
    with open(eval_output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] 评测结果已保存: {eval_output_file}")

    return output_data


def parse_range(range_str: str, max_val: int) -> tuple:
    if "-" in range_str:
        start, end = range_str.split("-")
        start = int(start) - 1
        end = int(end)
    else:
        idx = int(range_str) - 1
        start = idx
        end = idx + 1

    start = max(0, min(start, max_val))
    end = max(0, min(end, max_val))

    return start, end


def list_test_cases(config: AppConfig):
    test_cases = load_test_cases(config)

    if not test_cases:
        print("[INFO] 无测试用例")
        return

    print("\n测试用例列表:")
    print("-" * 80)
    for tc in test_cases:
        question_preview = tc["question"][:50] + "..." if len(tc["question"]) > 50 else tc["question"]
        print(f"[{tc['id']}] {question_preview}")


def list_log_turns(config: AppConfig):
    # 传递测试集文件给 LogParser，同时透传时间窗/轮次策略等新参数（审查 #12 env 契约）
    test_set_file = config.test_set.test_set_file if config.test_set.test_set_file else None
    parser = LogParser(
        config.log_parser.log_file,
        config.log_parser.target_dates,
        config.log_parser.target_keywords,
        target_questions=getattr(config.log_parser, "target_questions", None),
        test_set_file=test_set_file,
        target_time_after=config.log_parser.target_time_after,
        target_time_before=getattr(config.log_parser, "target_time_before", None),
        turn_boundary=getattr(config.log_parser, "turn_boundary", "intent_anchor"),
        parse_catalog_metadata=getattr(config.log_parser, "parse_catalog_metadata", True),
    )

    turns = parser.parse()

    if not turns:
        print("[INFO] 无日志记录")
        return

    print("\n日志轮次列表:")
    print("-" * 80)
    for i, turn in enumerate(turns):
        question_preview = (
            turn.user_question[:50] + "..."
            if len(turn.user_question) > 50
            else turn.user_question
        )
        print(f"[{i + 1}] {turn.date} {turn.question_time} - {question_preview}")
        print(f"    耗时: {turn.total_cost_ms}ms, API调用: {turn.api_call_count}次")


def main():
    parser = argparse.ArgumentParser(description="日志评测系统")
    parser.add_argument("--env", help="环境变量文件路径")
    parser.add_argument("--log", help="日志文件路径")
    parser.add_argument("--test-set", help="测试集文件路径")
    parser.add_argument("--mapping", help="手动映射文件路径")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--dates", help="日期过滤 (格式: 2025-07-10,2025-07-11)")
    parser.add_argument("--keywords", help="关键词过滤 (逗号分隔)")
    parser.add_argument("--turn-range", help="日志轮次范围 (格式: 1-10)")
    parser.add_argument("--test-range", help="测试用例范围 (格式: 1-10)")
    parser.add_argument("--list-tests", action="store_true", help="列出所有测试用例")
    parser.add_argument("--list-logs", action="store_true", help="列出所有日志轮次")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评测，仅解析日志")

    args = parser.parse_args()

    load_env_file(args.env)

    config = load_config(
        log_file=args.log,
        test_set_file=args.test_set,
        mapping_file=args.mapping,
        output_dir=args.output,
        dates=args.dates,
        keywords=args.keywords,
    )

    if args.list_tests:
        list_test_cases(config)
        return

    if args.list_logs:
        list_log_turns(config)
        return

    result = run_evaluation(
        config,
        turn_range=args.turn_range,
        test_range=args.test_range,
    )

    if result and not args.skip_eval:
        try:
            from report_generator import generate_html_report

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_file = os.path.join(
                config.output.output_dir, f"政务数据问答助手测评报告_{timestamp}.html"
            )
            generate_html_report(result, html_file)
            print(f"\n[INFO] HTML报告已生成: {html_file}")
        except ImportError:
            print("\n[WARN] report_generator 模块未找到，跳过HTML报告生成")


if __name__ == "__main__":
    main()
