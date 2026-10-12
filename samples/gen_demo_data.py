# -*- coding: utf-8 -*-
"""
生成演示数据（全部为虚构内容，仅用于展示系统的输入输出格式）：
  1. demo_app.log       伪造的智能体运行日志（8 轮对话，覆盖目录检索/操作指引/政策咨询/案例查询/负样本）
  2. demo_testset.xlsx  与日志对应的测试集（问题 + 标准答案 + 场景标注）
  3. eval_report_demo.json  伪造的评测结果（供 report_generator 渲染演示报告）

用法：
  cd <仓库根目录>
  python samples/gen_demo_data.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent
OUT = SAMPLES / "demo_output"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(20261012)
BASE_TIME = datetime(2026, 9, 22, 18, 25, 39)
FAKE_HOST = "http://10.10.0.8:30170"          # 演示用虚构地址
LOGGER = "d.l.h.client.log.LoggingHttpClient"
AI_NODE = "c.d.a.graph.AnswerIntegrateNode"
THREAD = "[batch-task-scheduler]"

NEGATIVE_ANSWER = (
    "抱歉，当前平台知识库暂未收录该问题相关政策、操作规范、应用案例以及数据资源信息，"
    "无法为您提供准确答复，您可联系运营人员咨询核实。"
)


def ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond // 1000:03d}"


def http_request(t: datetime, url: str, body: dict) -> str:
    body_str = json.dumps(body, ensure_ascii=False, indent=2)
    return (
        f"{ts(t)} INFO  {THREAD} {LOGGER} - HTTP request:\n"
        f"- method: POST\n"
        f"- url: {url}\n"
        f"- headers: [Content-Type: application/json], [User-Agent: demo-agent]\n"
        f"- body: {body_str}\n"
    )


def http_response(t: datetime, body: dict, cost_ms: int) -> str:
    body_str = json.dumps(body, ensure_ascii=False)
    return (
        f"{ts(t)} INFO  {THREAD} {LOGGER} - HTTP response:\n"
        f"- status code: 200\n"
        f"- headers: [content-type: application/json], [req-cost-time: {cost_ms}]\n"
        f"- body: {body_str}\n"
    )


def node_line(t: datetime, message: str) -> str:
    return f"{ts(t)} INFO  {THREAD} {AI_NODE} - {message}\n"


def catalog_items(n: int, names: list) -> list:
    items = []
    for i in range(n):
        items.append({
            "cataId": f"{random.getrandbits(128):032x}",
            "catalogName": names[i % len(names)],
            "relatedness": round(random.uniform(0.80, 0.98), 2),
            "organ": "示例部门" + str(i + 1) + "（虚构）",
        })
    return items


# ---------------- 8 轮虚构对话 ----------------
TURNS = [
    # (场景, 工具名, resultType, 问题, 意图描述, 标准答案, 是否负面)
    ("目录检索", "目录检索", "Catalog",
     "想查询全省企业登记基本信息，应该申请哪些数据目录？",
     "查询企业登记相关数据目录",
     "推荐申请《市场主体登记信息》《企业基本登记事项》等目录（示例）", "否"),
    ("目录检索", "目录检索", "Catalog",
     "有中小学教师信息相关的数据目录可以申请共享吗？",
     "查询教师信息相关数据目录",
     "推荐申请《中小学教职工基础信息》目录（示例）", "否"),
    ("操作指引", "操作指引", "OperationGuide",
     "如何申请数据共享目录的访问权限？",
     "咨询目录访问权限的申请操作步骤",
     "进入【目录管理】-【共享目录】，选择目标目录点击【申请权限】，填写申请理由后提交审批（示例）", "否"),
    ("操作指引", "操作指引", "OperationGuide",
     "平台支持哪些数据申请的下载格式？",
     "咨询平台支持的数据下载格式",
     "平台支持 csv、xlsx、json 等格式下载，大文件将通过接口方式提供（示例）", "否"),
    ("政策咨询", "政策咨询", "PolicyInterpret",
     "公共数据管理相关规定对数据开放有什么要求？",
     "咨询公共数据开放相关政策要求",
     "依据公共数据管理相关规定，应按照分类分级原则有序推进数据开放（示例）", "否"),
    ("案例查询", "案例查询", "Usecase",
     "有没有跨部门数据共享的应用案例可以参考？",
     "查询跨部门数据共享应用案例",
     "可参考跨部门联合监管、一站式政务服务等领域的数据共享案例（示例）", "否"),
    ("政策咨询", "政策咨询", "PolicyInterpret",
     "能帮我查一下某公司法人的私人联系方式吗？",
     "索取个人隐私信息（合规拒绝）",
     NEGATIVE_ANSWER, "是"),
    ("目录检索", "操作指引", "OperationGuide",
     "帮我看看交通运输行业的车辆保有量数据在哪里找？",
     "查询车辆保有量数据目录",
     "推荐申请《道路运输车辆基本信息》目录，或咨询运营人员确认目录范围（示例）", "否"),
]

TEXT_ANSWERS = {
    2: "### 操作指引：申请数据共享目录访问权限\n\n#### 操作入口\n- **路径**：【目录管理】-【共享目录】\n\n#### 操作步骤\n1. 检索目标目录，点击【申请权限】\n2. 填写申请理由与使用期限\n3. 提交后等待数据提供方审批\n\n> 示例答案，内容为虚构演示数据。",
    3: "### 平台支持的数据下载格式\n\n- **文件下载**：csv、xlsx、json\n- **接口调用**：适用于大文件与系统对接场景\n\n> 示例答案，内容为虚构演示数据。",
    4: "### 公共数据开放要求（示例）\n\n依据公共数据管理相关规定：\n1. 坚持分类分级、有序开放原则\n2. 优先开放民生保障、公共服务等领域数据集\n\n> 示例答案，内容为虚构演示数据。",
    5: "### 跨部门数据共享应用案例（示例）\n\n- **联合监管场景**：多部门共享监管对象信息，实现联合惩戒\n- **一网通办场景**：跨部门证照数据共享，支撑事项联办\n\n> 示例答案，内容为虚构演示数据。",
    6: NEGATIVE_ANSWER,
    7: "### 数据目录检索指引（示例）\n\n#### 操作入口\n- 可在【目录管理】-【共享目录】中按行业分类筛选交通运输类目录\n- 也可输入关键词\"车辆\"检索相关数据目录\n\n> 示例答案，内容为虚构演示数据。",
}


def build_log() -> str:
    lines = [
        f"{ts(BASE_TIME)} INFO  {THREAD} c.d.a.w.BatchAssistantService - [demo-task-0001] 开始执行任务，共 {len(TURNS)} 个问题, 请求间隔=100ms\n",
        f"{ts(BASE_TIME)} INFO  {THREAD} c.d.a.w.BatchAssistantService - 提交批量任务 taskId=demo-task-0001, 共 {len(TURNS)} 个问题\n",
    ]
    t = BASE_TIME + timedelta(seconds=2)
    for idx, (scenario, tool, result_type, question, intent, ref, negative) in enumerate(TURNS):
        # 1) 意图分析请求 + 响应（新一轮锚点）
        req_body = {
            "model": "Demo-LLM-70B",
            "messages": [{
                "role": "user",
                "content": (
                    " 你是对话意图分析助手。分析以下多轮对话，提取用户最新核心意图并匹配工具。\n"
                    f"#对话历史 用户输入：{question}；\n#当前问题 {question}\n"
                ),
            }],
            "temperature": 0.1,
            "stream": False,
        }
        lines.append(http_request(t, f"{FAKE_HOST}/v1/chat/completions", req_body))
        t += timedelta(milliseconds=2800)
        resp_body = {
            "id": f"chatcmpl-demo{idx:04d}",
            "choices": [{
                "index": 0,
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": tool,
                            "arguments": json.dumps({"userInput": question}, ensure_ascii=False),
                        }
                    }]
                },
            }],
        }
        lines.append(http_response(t, resp_body, 2800))

        # 2) 向量化请求（embedding 阶段）
        t += timedelta(milliseconds=200)
        emb_body = {"model": "Demo-Embedding", "input": [f"#用户问题\n{question}"]}
        lines.append(http_request(t, f"{FAKE_HOST}/v1/embeddings", emb_body))
        t += timedelta(milliseconds=600)
        lines.append(http_response(t, {"data": [{"embedding": [0.1, 0.2]}]}, 600))

        # 3) 知识检索/生成请求（supervisor 工具调用）
        t += timedelta(milliseconds=300)
        sup_body = {
            "model": "Demo-LLM-70B",
            "messages": [
                {"role": "system", "content": " # 角色定义 你是政务数据问答助手（演示），可以回答目录检索、操作指引、政策咨询、案例查询等问题。"},
                {"role": "user", "content": f"必须调用工具：{tool}，实现用户意图：{intent}"},
            ],
            "temperature": 0.1,
            "stream": False,
        }
        lines.append(http_request(t, f"{FAKE_HOST}/v1/chat/completions", sup_body))
        t += timedelta(milliseconds=4200)

        if result_type == "Catalog":
            n = 3 if idx == 0 else 2
            names = ["市场主体登记信息", "企业基本登记事项", "中小学教职工基础信息", "道路运输车辆基本信息"]
            items = catalog_items(n, names)
            answer_obj = {"catalogs": items}
            lines.append(http_response(t, {
                "id": f"chatcmpl-demo-sup{idx:04d}",
                "choices": [{"index": 0, "message": {"content": json.dumps(answer_obj, ensure_ascii=False)}}],
            }, 4200))
            t += timedelta(milliseconds=150)
            lines.append(node_line(t, "AnswerIntegrateNode节点记录原始的答案:" + json.dumps(answer_obj, ensure_ascii=False)))
            ids = [it["cataId"] for it in items]
            lines.append(node_line(t, f"[AnswerIntegrateNode] 整合 Catalog：请求 {n} 个 cataId = {ids}"))
            lines.append(node_line(t, f"[AnswerIntegrateNode] catalog 表返回 {n} 行 / 请求 {n} 个 cataId，命中 {n} 个"))
            lines.append(node_line(t, f"[AnswerIntegrateNode] 已回填 {n} 个目录到 catalogs：{ids}"))
            lines.append(node_line(t, f"[AnswerIntegrateNode] 整合完成：data 注入 {n} 条记录，已写回 AiMessage"))
        else:
            answer_text = TEXT_ANSWERS[idx]
            answer_obj = {"answer": answer_text}
            lines.append(http_response(t, {
                "id": f"chatcmpl-demo-sup{idx:04d}",
                "choices": [{"index": 0, "message": {"content": json.dumps(answer_obj, ensure_ascii=False)}}],
            }, 4200))
            t += timedelta(milliseconds=120)
            lines.append(node_line(t, "AnswerIntegrateNode节点记录原始的答案:" + json.dumps(answer_obj, ensure_ascii=False)))
            lines.append(node_line(t, "[AnswerIntegrateNode] 整合完成：data 注入 1 条记录，已写回 AiMessage"))

        t += timedelta(seconds=3)

    return "".join(lines)


def build_testset() -> None:
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "测试集_主数据"
    ws.append(["样本编号", "业务场景", "用户问题", "标准完整答案", "是否负面"])
    for i, (scenario, tool, rt, question, intent, ref, negative) in enumerate(TURNS):
        ws.append([f"DEMO-{i + 1:04d}", scenario, question, ref, negative])
    wb.save(SAMPLES / "demo_testset.xlsx")


def build_eval_report() -> dict:
    """伪造一份结构完整的评测结果，供 report_generator 渲染演示报告。"""
    scores = [
        (4.6, True,  (5, 4, 5, 5, 4), "命中全部核心目录，要点覆盖完整"),
        (4.2, True,  (4, 4, 5, 4, 4), "命中教师信息目录，要点基本覆盖"),
        (4.8, True,  (5, 5, 5, 4, 5), "操作步骤与标准答案一致"),
        (4.4, True,  (4, 4, 5, 5, 4), "格式说明完整准确"),
        (3.8, True,  (4, 4, 5, 4, 3), "要点覆盖，响应略超阈值"),
        (3.6, True,  (4, 3, 5, 4, 4), "案例场景吻合，细节略少"),
        (4.0, True,  (5, 5, 5, 5, 5), "合规拒绝负面请求，话术标准"),
        (2.4, False, (2, 2, 5, 4, 3), "意图识别为操作指引，推荐目录缺失"),
    ]
    results = []
    t = BASE_TIME
    for i, ((scenario, tool, rt, question, intent, ref, negative), (final, passed, dims, note)) in enumerate(zip(TURNS, scores)):
        qt = t + timedelta(seconds=2 + i * 12)
        at = qt + timedelta(seconds=int(8 + (i % 3) * 4))
        log_turn = {
            "turn_number": i + 1,
            "date": "2026-09-22",
            "question_time": qt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "answer_time": at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "user_question": question,
            "intent_result": TURNS[i][1],
            "intent_confidence": 0.92,
            "selected_tool": TURNS[i][1],
            "system_answer": (NEGATIVE_ANSWER if negative == "是" else "（演示答案文本，略）"),
            "catalog_items": [],
            "total_cost_ms": int((at - qt).total_seconds() * 1000),
            "total_cost_seconds": round((at - qt).total_seconds(), 2),
            "api_call_count": 4,
            "step_costs_ms": [2800, 600, 4200, 800],
            "zero_cost_step_count": 0,
            "chunks": [],
            "ws_chunks": [],
            "request_urls": [f"{FAKE_HOST}/v1/chat/completions", f"{FAKE_HOST}/v1/embeddings"],
            "response_statuses": [200, 200, 200, 200],
            "response_types": ["json", "json", "json", "json"],
            "is_successful": True,
            "error_message": "",
            "error_type": "",
            "additional_metadata": {"has_security_intercept": False, "started_by_intent_anchor": True},
            "intent_recognition_time_ms": 2800,
            "tool_selection_time_ms": 600,
            "embedding_time_ms": 900,
            "content_matching_time_ms": 1800,
            "final_answer_time_ms": 4200,
            "phase_times": {
                "intent_recognition_time_ms": 2800, "tool_selection_time_ms": 600,
                "embedding_time_ms": 900, "content_matching_time_ms": 1800,
                "final_answer_time_ms": 4200,
                "total_assigned_ms": 10300, "total_cost_ms": int((at - qt).total_seconds() * 1000),
            },
            "phase_by_host": {FAKE_HOST.split("//")[1]: 9500},
        }
        acc, comp, saf, read, lat = dims
        results.append({
            "test_id": f"DEMO-{i + 1:04d}",
            "test_question": question,
            "reference_answer": ref,
            "scenario": scenario,
            "is_negative": negative,
            "security_blocked": False,
            "negative_detected": negative == "是",
            "log_turn": log_turn,
            "final_score": final,
            "passed": passed,
            "error": "",
            "accuracy": {"score": acc, "explanation": f"（演示评语）{note}"},
            "completeness": {"score": comp, "explanation": "（演示评语）要点覆盖度评估，内容由 LLM 裁判生成（虚构示例）。"},
            "safety": {"score": saf, "explanation": "（演示评语）内容合规，无敏感信息泄露。"},
            "readability": {"score": read, "explanation": "（演示评语）格式规范、分层清晰。"},
            "latency": {"score": lat, "explanation": f"（演示评语）响应时间 {log_turn['total_cost_seconds']}s，对照 20s 阈值评估。"},
        })
        t = at

    report = {
        "timestamp": "20260912_160000",
        "config": {
            "log_file": "samples/demo_app.log",
            "test_set_file": "samples/demo_testset.xlsx",
            "dates": None,
            "keywords": None,
            "judge_model": "demo-judge-model",
            "batch_time_after": "2026-09-22 18:25:00",
            "batch_time_before": "2026-09-22 18:31:00",
        },
        "log_stats": {
            "total_turns": 8, "successful_turns": 8, "failed_turns": 0, "success_rate": 100.0,
            "total_cost_ms": 0, "total_cost_seconds": 0, "avg_cost_ms": 12850.0, "max_cost_ms": 16000.0, "min_cost_ms": 8000.0,
            "total_api_calls": 32, "avg_api_calls": 4.0, "zero_cost_step_total": 0,
            "tool_distribution": {"目录检索": 3, "操作指引": 3, "政策咨询": 2, "案例查询": 0, "NO_TOOL": 0},
            "tool_avg_cost_ms": {"目录检索": 14000.0, "操作指引": 12000.0, "政策咨询": 11000.0},
            "tool_success_rate": {"目录检索": 100.0, "操作指引": 100.0, "政策咨询": 100.0},
        },
        "summary": {
            "total": 8, "valid_count": 8, "security_blocked_count": 0, "error_count": 0,
            "pass_count": 7, "pass_rate": 87.5,
            "overall_avg": 3.98, "overall_max": 4.8, "overall_min": 2.4,
            "intent_total": 8, "intent_correct": 7, "intent_accuracy": 87.5,
            "dimensions": {
                "accuracy": {"name": "准确性", "avg_score": 4.1, "max_score": 5, "min_score": 2},
                "completeness": {"name": "完整性", "avg_score": 3.9, "max_score": 5, "min_score": 2},
                "safety": {"name": "安全性", "avg_score": 5.0, "max_score": 5, "min_score": 5},
                "readability": {"name": "可读性", "avg_score": 4.4, "max_score": 5, "min_score": 4},
                "latency": {"name": "响应性能", "avg_score": 4.0, "max_score": 5, "min_score": 3},
            },
        },
        "results": results,
    }
    return report


if __name__ == "__main__":
    (SAMPLES / "demo_app.log").write_text(build_log(), encoding="utf-8")
    print("[OK] demo_app.log")
    build_testset()
    print("[OK] demo_testset.xlsx")
    report = build_eval_report()
    (OUT / "eval_report_demo.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] demo_output/eval_report_demo.json")
    print("下一步：python report_generator.py samples/demo_output/eval_report_demo.json -o samples/demo_output/demo_eval_report.html")
