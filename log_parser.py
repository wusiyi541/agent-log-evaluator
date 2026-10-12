import re
import json
import os
import sys
from difflib import SequenceMatcher
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse
from collections import defaultdict

# ANSI 转义码（部分日志会带颜色控制符，如时间戳与日志级别之间的 \x1b[34mINFO\x1b[0m），
# 会破坏"时间戳+日志级别"连续的行判定，需在解析前剥离。
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _strip_ansi(text: str) -> str:
    """剥离文本中的 ANSI/OSC 转义序列，避免颜色控制符干扰日志行识别。"""
    return _OSC_RE.sub("", _ANSI_RE.sub("", text))


# 目录富化元数据：从日志的"已回填"或"富化目录明细"行提取 cataId → catalogName
_CATALOG_PAIR_RE = re.compile(r'([0-9a-f]{32})=>([^,\]]+)')
_CATALOG_NAME_RE = re.compile(
    r'\\"cataId\\"\s*:\s*\\"([0-9a-f]{32})\\"[^}]*?\\"catalogName\\"\s*:\s*\\"([^\\"]+)\\"',
    re.S,
)
_CATALOG_NAME_RE2 = re.compile(
    r'\\"catalogName\\"\s*:\s*\\"([^\\"]+)\\"[^}]*?\\"cataId\\"\s*:\s*\\"([0-9a-f]{32})\\"',
    re.S,
)

# === 共享的 timestamp 正则（P2 审查 #5：三处不再各写一套互不兼容版本）===
# 通用 ISO8601 + 空格版 + 秒级空格版，一个元组覆盖：
#   group(1) = YYYY-MM-DD
#   group(2) = 时间部分（可能含毫秒/TZ）
_TS_F_ISO = r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)'
_TS_F_MS  = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}\.\d+)'
_TS_F_SEC = r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(?![.\d])'

# parse() 用于"整行匹配起始时间戳"（line.match）
_TIMESTAMP_PATTERNS_LINE = tuple(
    re.compile('^' + p) for p in (_TS_F_ISO, _TS_F_MS, _TS_F_SEC)
)

# _parse_request / _parse_response 用于"块内 re.search 抽 timestamp"（不锚定 ^，且只需要完整字符串即可，无分组需求）
_TIMESTAMP_PATTERNS_BLOCK = (
    re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})'),
    re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+'),
    re.compile(r'(?<![.\d])\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?![.\d])'),
)

# 意图分析助手锚点：一条含"对话意图分析助手"的请求 = 一次用户提问的开始
_INTENT_ANCHOR_RE = re.compile(r"对话意图分析助手")


@dataclass
class CatalogItem:
    """目录检索 / 案例查询 返回的目录项（带中文名称富化）"""
    cataId: str = ""
    catalogName: str = ""
    relatedness: float = 0.0
    organ: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"cataId": self.cataId, "catalogName": self.catalogName}
        if self.relatedness: d["relatedness"] = self.relatedness
        if self.organ: d["organ"] = self.organ
        if self.extra: d.update(self.extra)
        return d


@dataclass
class ChunkInfo:
    chunk_id: str = ""
    content: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogTurn:
    turn_number: int  # 无默认，必须第一组
    date: str
    question_time: str
    answer_time: str
    user_question: str
    intent_result: str
    intent_confidence: float
    selected_tool: str
    system_answer: Union[str, List[Dict[str, Any]]]  # 文本 或 Catalog/UseCase 富化清单（答案唯一权威字段）
    total_cost_ms: float
    total_cost_seconds: float
    api_call_count: int
    step_costs_ms: List[float]
    # --- 以下字段均有默认值，必须在"无默认字段"之后 ---
    catalog_items: List[CatalogItem] = field(default_factory=list)  # P0 2.1：目录名富化
    zero_cost_step_count: int = 0  # P1 1.3：cost_time_ms==0 的步骤数
    chunks: List[ChunkInfo] = field(default_factory=list)
    ws_chunks: List[Dict[str, Any]] = field(default_factory=list)  # P0：WS检索元数据(前端结构)，扁平列表，每项含 scene/text/score/embeddingId/source/metadata
    request_urls: List[str] = field(default_factory=list)
    response_statuses: List[str] = field(default_factory=list)
    is_successful: bool = True
    error_message: str = ""
    error_type: str = ""  # P2 3.5：归一化后的错误类型
    response_types: List[str] = field(default_factory=list)
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    intent_recognition_time_ms: float = 0.0
    tool_selection_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    content_matching_time_ms: float = 0.0
    final_answer_time_ms: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)
    phase_by_host: Dict[str, Dict[str, float]] = field(default_factory=dict)  # P2 1.4

    def to_dict(self) -> Dict[str, Any]:
        sys_ans: Any = self.system_answer
        if isinstance(sys_ans, list):
            sys_ans_out = sys_ans
        else:
            sys_ans_out = sys_ans
        return {
            "turn_number": self.turn_number,
            "date": self.date,
            "question_time": self.question_time,
            "answer_time": self.answer_time,
            "user_question": self.user_question,
            "intent_result": self.intent_result,
            "intent_confidence": self.intent_confidence,
            "selected_tool": self.selected_tool,
            "system_answer": sys_ans_out,
            "catalog_items": [c.to_dict() for c in self.catalog_items],
            "total_cost_ms": self.total_cost_ms,
            "total_cost_seconds": self.total_cost_seconds,
            "api_call_count": self.api_call_count,
            "step_costs_ms": self.step_costs_ms,
            "zero_cost_step_count": self.zero_cost_step_count,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.content[:200],
                    "score": c.score,
                    "metadata": c.metadata,
                }
                for c in self.chunks
            ],
            "ws_chunks": self.ws_chunks,
            "request_urls": self.request_urls,
            "response_statuses": self.response_statuses,
            "is_successful": self.is_successful,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "response_types": self.response_types,
            "additional_metadata": self.additional_metadata,
            "intent_recognition_time_ms": self.intent_recognition_time_ms,
            "tool_selection_time_ms": self.tool_selection_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "content_matching_time_ms": self.content_matching_time_ms,
            "final_answer_time_ms": self.final_answer_time_ms,
            "phase_times": self.phase_times,
            "phase_by_host": self.phase_by_host,
        }


class LogParser:
    def __init__(
        self,
        log_file: str,
        target_dates: Optional[set] = None,
        target_keywords: Optional[List[str]] = None,
        target_questions: Optional[List[str]] = None,
        test_set_file: Optional[str] = None,
        target_time_after: Optional[str] = None,
        target_time_before: Optional[str] = None,
        turn_boundary: str = "intent_anchor",
        parse_catalog_metadata: bool = True,
    ):
        """
        Args:
            target_questions: 问题白名单；任意白名单项是问题的子串/完全相等即保留（实现见 _should_include）。
            target_time_after / target_time_before: 双端闭区间 [after, before]，支持 ISO8601/空格分隔
                （毫秒/时区可选），传 '2026-08-19 16:33:01.602' 会按真实 datetime 精确比较，不会被截到秒误判。
            turn_boundary: 轮次分界策略（P0 1.1）
              - "intent_anchor"（默认）：以含「对话意图分析助手」的请求为锚点，
                每条用户提问 = 1 个 turn，绝不进行相似度合并。
              - "test_set"     ：以测试集问题相似度为分界（原 merge 逻辑）。
              - "similarity"   ：旧的相似度+时间差合并逻辑（兜底用）。
            parse_catalog_metadata: 是否预扫日志构建 cataId→catalogName 映射（P0 2.1）。
        """
        self.log_file = log_file
        self.target_dates = target_dates
        self.target_keywords = target_keywords
        self.target_questions = target_questions
        self.test_set_file = test_set_file
        self.target_time_after = target_time_after
        self.target_time_before = target_time_before

        valid_strategies = {"intent_anchor", "test_set", "similarity"}
        if turn_boundary not in valid_strategies:
            print(f"[WARN] turn_boundary='{turn_boundary}' 非法，回退 'intent_anchor'")
            turn_boundary = "intent_anchor"
        self.turn_boundary = turn_boundary

        self.parse_catalog_metadata = parse_catalog_metadata
        # strict_noise_filter 已按 #1 审查移除（空实现无分支影响，属欺骗接口）；如需要在 _is_valid_question 加实现。

        self.test_set_questions: List[str] = []
        self.turns: List[LogTurn] = []
        self._raw_turns_data: List[Dict] = []

        # P0 2.1：目录名映射（cataId → 中文名），可能从 AnswerIntegrateNode 的"已回填"日志来
        self.catalog_name_map: Dict[str, str] = {}
        self._turn_intent_anchor_flags: List[bool] = []  # 并行于 raw_turns，标记每 turn 边界是否来自"意图分析锚点"

        # P2 3.7：pandas 软依赖 — ImportError 仅打印 WARN，不崩溃
        if test_set_file:
            self._load_test_set(test_set_file)

        # P0 2.1：主动扫描日志一次，构建目录名映射（很快，O(lines) 一次 for）
        if parse_catalog_metadata and os.path.exists(log_file):
            self._preload_catalog_metadata(log_file)
    
    def _load_test_set(self, test_set_file: str):
        """加载测试集问题列表"""
        try:
            import pandas as pd
            
            # 读取所有工作表，找到包含问题列的工作表
            xls = pd.ExcelFile(test_set_file)
            df = None
            
            for sheet_name in xls.sheet_names:
                sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
                for col in sheet_df.columns:
                    if '用户问题' in str(col) or '问题' == str(col):
                        df = sheet_df
                        break
                if df is not None:
                    break
            
            if df is None:
                # 如果没找到，尝试用openpyxl直接加载
                from openpyxl import load_workbook
                wb = load_workbook(test_set_file, read_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
                    if any('用户问题' in str(h) or '问题' == str(h) for h in headers):
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        break
                wb.close()
            
            if df is None:
                print(f"[WARN] 测试集中未找到包含问题列的工作表")
                return
            
            # 尝试找到问题列
            question_col = None
            for col in df.columns:
                if '用户问题' in str(col) or '问题' == str(col):
                    question_col = col
                    break
            
            if question_col is None:
                # 找包含"问"的列
                for col in df.columns:
                    if '问' in str(col):
                        question_col = col
                        break
            
            if question_col:
                self.test_set_questions = df[question_col].dropna().tolist()
                print(f"[INFO] 已加载测试集，共 {len(self.test_set_questions)} 条用户提问")
            else:
                print(f"[WARN] 测试集中未找到问题列，可用列: {df.columns.tolist()}")
                
        except ImportError:
            print("[WARN] pandas / openpyxl 未安装，无法加载测试集，请 pip install pandas openpyxl")
        except Exception as e:
            print(f"[WARN] 加载测试集失败: {e}")

    # ------------------------------------------------------------------
    # P0 2.1 / 2.2：目录富化元数据预解析 + 富化后的最终答案提取
    # ------------------------------------------------------------------
    def _preload_catalog_metadata(self, log_path: str) -> None:
        """扫一轮日志，从"已回填 catalogs：xxx"或"富化目录明细"行抽取 cataId→catalogName。"""
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "已回填" in line and "catalogs" in line:
                        for cid, name in _CATALOG_PAIR_RE.findall(line):
                            name = name.strip()
                            if cid and name and cid not in self.catalog_name_map:
                                self.catalog_name_map[cid] = name
                    # AnswerIntegrateNode 可能打印 JSON 格式的回填明细
                    if "富化目录明细" in line or "catalogName" in line:
                        for cid, name in _CATALOG_NAME_RE.findall(line):
                            name = name.strip()
                            if cid and name and cid not in self.catalog_name_map:
                                self.catalog_name_map[cid] = name
                        for name, cid in _CATALOG_NAME_RE2.findall(line):
                            name = name.strip()
                            if cid and name and cid not in self.catalog_name_map:
                                self.catalog_name_map[cid] = name
        except FileNotFoundError:
            print(f"[WARN] _preload_catalog_metadata 文件不存在: {log_path}")
        except Exception as e:
            print(f"[WARN] _preload_catalog_metadata 失败: {e}")

    def classify_error_type(self, turn: "LogTurn") -> str:
        """P2 3.5：将错误信息归一化到 6 类之一；成功轮次（非安全拦截）返回空字符串。"""
        # 先处理安全拦截（优先级最高，即使 is_successful 被误伤）
        if turn.additional_metadata.get("has_security_intercept"):
            return "安全拦截"
        # 成功轮次：无错误类型
        if turn.is_successful:
            return ""

        msg = (turn.error_message or "") + " " + str(turn.system_answer or "")
        statuses = [str(x) for x in (turn.response_statuses or [])]
        if "Milvus" in msg or "vector" in msg.lower() or "collection" in msg.lower():
            return "向量库异常"
        if "Embedding" in msg or "embedding" in msg or "text-embedding" in msg:
            return "Embedding服务异常"
        if "timeout" in msg.lower() or "超时" in msg:
            return "接口超时"
        if any(s in ("500", "502", "503", "504") for s in statuses):
            return "服务端5xx"
        if not turn.system_answer and not turn.catalog_items:
            return "空回答"
        return "其他"

    def _is_user_question(self, question: str, threshold: float = 0.9) -> bool:
        """判断一个问题是否是测试集中的用户提问
        
        使用高阈值(0.9)确保只有非常相似的才是真正的用户提问
        相似度在0.6-0.9之间的可能是机器改写，会被合并到前一个用户提问
        """
        if not self.test_set_questions:
            return False
        
        for test_q in self.test_set_questions:
            # 使用更高的阈值，确保只有几乎相同的才是用户提问
            similarity = self._calculate_similarity(question, str(test_q))
            if similarity >= threshold:
                return True
        
        return False

    def parse(self) -> List[LogTurn]:
        if not os.path.exists(self.log_file):
            print(f"[ERROR] 日志文件不存在: {self.log_file}")
            return []

        print(f"[INFO] 正在读取日志文件: {self.log_file}")
        print(f"[INFO] 过滤条件: 日期={self.target_dates}, 时间范围=[{self.target_time_after or '-∞'} .. {self.target_time_before or '+∞'}], 关键词={self.target_keywords}")

        with open(self.log_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 剥离 ANSI 颜色转义码（新格式日志在「时间戳 + 日志级别」间插入了 \x1b[34m 等控制符）
        content = _strip_ansi(content)

        lines = content.split("\n")
        print(f"[INFO] 日志总行数: {len(lines)}")

        # 变更 #2：预扫描 AnswerIntegrateNode 答案标记（节点记录原始的答案 / 已回填），供后续按时间窗关联到 turn
        self._ai_markers: List[Dict[str, Any]] = self._extract_ai_markers(lines)
        print(f"[INFO] AnswerIntegrateNode 答案标记数: {len(self._ai_markers)}")

        # 变更 #4：预扫描 [召回片段-<场景>-WS格式] 检索元数据（前端结构），供报告展示
        self._ws_chunks_raw: List[Dict[str, Any]] = self._extract_ws_chunks(lines)
        _ws_total = sum(len(e["chunks"]) for e in self._ws_chunks_raw)
        print(f"[INFO] WS检索元数据(前端结构) 块数: {len(self._ws_chunks_raw)}，检索片段总数: {_ws_total}")

        timestamp_indices = []
        stop_parsing = False  # 审查 #3：时间戳单调递增，超过 target_time_before 就不再看后面所有行
        after_dt = self._parse_ts(self.target_time_after) if self.target_time_after else None
        before_dt = self._parse_ts(self.target_time_before) if self.target_time_before else None

        for i, line in enumerate(lines):
            if stop_parsing:
                break
            for pattern in _TIMESTAMP_PATTERNS_LINE:
                m = pattern.match(line)
                if m:
                    date = m.group(1)
                    if len(m.groups()) >= 2:
                        time_part = m.group(2)
                        timestamp = f"{date}T{time_part}" if "T" in line else f"{date} {time_part}"
                    else:
                        timestamp = date
                    if self.target_dates is None or date in self.target_dates:
                        # 审查 #4：使用 datetime 双端闭区间比较（支持毫秒/TZ 传参的统一格式）
                        ts_dt = self._parse_ts(timestamp)
                        if after_dt and (ts_dt is None or ts_dt < after_dt):
                            break
                        if before_dt and (ts_dt is None or ts_dt > before_dt):
                            # 审查 #3：时间戳单调递增 → 后面所有行都必然 > before，直接跳过剩余内容
                            stop_parsing = True
                            break
                        timestamp_indices.append((i, date, timestamp))
                    break

        print(f"[INFO] 匹配到的时间戳条目数: {len(timestamp_indices)}")

        blocks = []
        for idx, (start_idx, _, _) in enumerate(timestamp_indices):
            if idx + 1 < len(timestamp_indices):
                end_idx = timestamp_indices[idx + 1][0]
            else:
                end_idx = len(lines)
            block_text = "\n".join(lines[start_idx:end_idx])
            blocks.append(block_text)

        request_blocks = []
        response_blocks = []

        for block in blocks:
            if "HTTP request:" in block:
                request_blocks.append(block)
            elif "HTTP response:" in block:
                response_blocks.append(block)

        print(f"[INFO] 请求块: {len(request_blocks)}, 响应块: {len(response_blocks)}")

        parsed_requests = [self._parse_request(b) for b in request_blocks]
        parsed_responses = [self._parse_response(b) for b in response_blocks]

        print(f"[INFO] 有用户问题的请求: {sum(1 for r in parsed_requests if r.get('user_question'))}")
        print(f"[INFO] 有响应类型的响应: {sum(1 for r in parsed_responses if r.get('response_type'))}")
        print(f"[INFO] 最终答案响应: {sum(1 for r in parsed_responses if r.get('response_type') == 'final_answer')}")

        all_events = []
        for r in parsed_requests:
            all_events.append(("req", r))
        for r in parsed_responses:
            all_events.append(("resp", r))

        all_events.sort(key=lambda x: x[1]["timestamp"])

        turns_data = []
        current_turn = {
            "question": None,
            "question_time": None,
            "question_date": None,
            "steps": [],
            "urls": [],
            "statuses": [],
            "response_types": [],
            "has_error": False,
            "error_message": "",
        }
        pending_request = None

        # ------------------------------------------------------------------
        # P0 1.1：按 turn_boundary 策略划分 turn
        #   intent_anchor（默认）：以 is_intent_anchor=True 的 req 作为新 turn 开始；
        #                         其它请求都并入当前 turn。避免"相似度过合并"导致 58 → 210 的损失。
        #   test_set            ：沿用旧逻辑（每条 user_question 都可能开 turn + 后续 merge_similar）
        #   similarity          ：旧相似度合并逻辑（兜底）
        # ------------------------------------------------------------------
        strategy = self.turn_boundary
        anchor_flags: List[bool] = []  # 与 turns_data 一一对应，记录本轮是否以"意图锚点"起始

        for event_type, data in all_events:
            if event_type == "req":
                question = data.get("user_question", "")
                is_anchor = bool(data.get("is_intent_anchor", False))

                # 意图锚点请求：即使 question 为空，也作为新 turn 开始（question 从后续步骤回补）
                should_open_new = False
                if strategy == "intent_anchor":
                    should_open_new = is_anchor
                else:
                    # test_set / similarity 模式：沿用原有逻辑，任何带有效问题的 req 都开 turn
                    should_open_new = bool(question)

                if should_open_new:
                    # 提交旧 turn（含数据时）
                    if current_turn["question"] and current_turn["steps"]:
                        if self._should_include(current_turn["question"]):
                            turns_data.append(current_turn.copy())
                            anchor_flags.append(bool(current_turn.get("__started_by_anchor")))
                    # 开新 turn
                    current_turn = {
                        "question": question,
                        "question_time": data.get("timestamp") or "",
                        "question_date": data.get("date") or "",
                        "steps": [],
                        "urls": [],
                        "statuses": [],
                        "response_types": [],
                        "has_error": False,
                        "error_message": "",
                        "__started_by_anchor": is_anchor,
                    }
                    pending_request = data
                else:
                    # intent_anchor 模式：锚点开 turn 后，后续 req（无锚点）都带同 question 继承，
                    # 若 current_turn 还没 question（锚点请求 body 空），从当前 req 回补问题。
                    if not current_turn["question"] and question:
                        current_turn["question"] = question
                        current_turn["question_time"] = data.get("timestamp") or current_turn["question_time"]
                        current_turn["question_date"] = data.get("date") or current_turn["question_date"]
                        current_turn["__started_by_anchor"] = current_turn.get("__started_by_anchor", False)
                    pending_request = data
            elif event_type == "resp":
                if pending_request:
                    step = {"request": pending_request, "response": data}
                    current_turn["steps"].append(step)
                    current_turn["urls"].append(pending_request.get("url", ""))
                    current_turn["statuses"].append(data.get("status", ""))

                    response_type = data.get("response_type", "")
                    current_turn["response_types"].append(response_type)

                    if "error" in response_type or data.get("status") != "200":
                        current_turn["has_error"] = True
                        current_turn["error_message"] = data.get("error_message", "")

                    # intent_anchor 模式：锚点请求是 chat/completions，question 可能为空，
                    # 所以在它的响应 answer 里找 userIntent / 对话内容里的 user question 回填。
                    if (
                        strategy == "intent_anchor"
                        and not current_turn["question"]
                        and response_type in ("intent", "tool_call", "final_answer")
                    ):
                        fallback_q = (
                            data.get("user_intent")
                            or (isinstance(data.get("answer"), str) and data["answer"] and len(data["answer"]) < 200
                                and self._is_valid_question(data["answer"]) and data["answer"])
                            or ""
                        )
                        if fallback_q:
                            current_turn["question"] = fallback_q

                    pending_request = None

        if current_turn["question"] and current_turn["steps"]:
            if self._should_include(current_turn["question"]):
                turns_data.append(current_turn.copy())
                anchor_flags.append(bool(current_turn.get("__started_by_anchor")))

        # 保存原始数据
        self._raw_turns_data = turns_data
        self._turn_intent_anchor_flags = anchor_flags

        print(
            f"[INFO] 识别到 {len(turns_data)} 个符合条件的对话轮次 "
            f"(turn_boundary={strategy}, 锚点起始={sum(anchor_flags)})"
        )

        # ---- 变更 #3：以 AnswerIntegrateNode 标记本身为锚点，做"最近前置提问"一对一关联 ----
        # 新日志格式下，标记时间戳晚于其所在轮次的提问，但可能早于"下一轮提问"，
        # 旧的时间窗 [qt, next_q) 关联会导致答案被相邻轮"截胡"（错位 1 步）。
        # 改为：每个标记归属到"提问时间 ≤ 标记时间 且 最接近"的那一轮（最近前置提问），
        # 每个轮次最多绑定一个标记（按标记时间升序分配），实现干净的一对一配对。
        self._turn_ai_marker = self._assign_ai_markers_to_turns(turns_data)

        # 变更 #4：将 WS检索元数据 按时间戳归属到对应轮次（复用最近前置提问逻辑）
        self._assign_ws_chunks_to_turns(turns_data)

        for i, turn_data in enumerate(turns_data):
            chunks: List[ChunkInfo] = []
            all_costs: List[float] = []
            zero_cost_step_count = 0
            final_answer: Union[str, List[Dict[str, Any]]] = ""
            catalog_items: List[CatalogItem] = []
            final_answer_time = ""
            intent_result = ""
            intent_confidence = 0.0
            selected_tool = ""
            response_types_list = []
            has_final_answer = False
            has_error = turn_data["has_error"]
            error_message = turn_data["error_message"]
            has_security_intercept = False
            # P2 1.4：分 host 统计 phase
            phase_by_host: Dict[str, Dict[str, float]] = defaultdict(
                lambda: {
                    "intent_recognition_time_ms": 0.0,
                    "tool_selection_time_ms": 0.0,
                    "embedding_time_ms": 0.0,
                    "content_matching_time_ms": 0.0,
                    "final_answer_time_ms": 0.0,
                    "total_cost_ms": 0.0,
                    "call_count": 0,
                    "fail_count": 0,
                }
            )

            for step in turn_data["steps"]:
                req = step["request"]
                resp = step["response"]
                cost = resp.get("cost_time_ms", 0) or 0
                try:
                    cost_f = float(cost)
                except (TypeError, ValueError):
                    cost_f = 0.0
                if cost_f <= 0:
                    zero_cost_step_count += 1
                all_costs.append(cost_f)

                host = req.get("host") or ""
                # 用 _classify_phase_for_step 对每步分 phase，便于 per-host 累积
                phase_key = self._classify_step_phase(step, turn_data["steps"])
                if host:
                    phase_by_host[host][phase_key] = phase_by_host[host].get(phase_key, 0.0) + cost_f
                    phase_by_host[host]["total_cost_ms"] = phase_by_host[host].get("total_cost_ms", 0.0) + cost_f
                    phase_by_host[host]["call_count"] = phase_by_host[host].get("call_count", 0) + 1
                    if resp.get("status") not in (None, "200", 200, ""):
                        phase_by_host[host]["fail_count"] = phase_by_host[host].get("fail_count", 0) + 1

                response_type = resp.get("response_type", "")
                response_types_list.append(response_type)

                if response_type == "security_intercept":
                    has_security_intercept = True
                    error_message = "安全拦截: " + resp.get("answer", "").split("\n")[0] if resp.get("answer") else "安全拦截"

                resp_intent = resp.get("user_intent", "")
                resp_tool = resp.get("tool", "")
                resp_confidence = resp.get("intent_confidence", 0.0)

                # 每看到一次 response_type=intent → 直接覆写（支持二次意图纠正）
                if response_type == "intent":
                    if resp_intent:
                        intent_result = resp_intent
                    if resp_tool:
                        selected_tool = resp_tool
                    if resp_confidence:
                        try:
                            intent_confidence = float(resp_confidence)
                        except (ValueError, TypeError):
                            pass
                else:
                    # 非 intent 响应：保持首次非空（只在缺省时回退写）
                    if resp_intent and not intent_result:
                        intent_result = resp_intent
                    if resp_tool and not selected_tool:
                        selected_tool = resp_tool
                    if resp_confidence and not intent_confidence:
                        try:
                            intent_confidence = float(resp_confidence)
                        except (ValueError, TypeError):
                            pass

                req_tool = req.get("tool", "")
                req_user_intent = req.get("user_intent", "")
                if req_tool and not selected_tool:
                    selected_tool = req_tool
                if req_user_intent and not intent_result:
                    intent_result = req_user_intent

                # 提取最终答案（优先 response_type=final_answer，富化目录名）
                if response_type == "final_answer":
                    final_answer = resp.get("final_answer", "")
                    # 若 final_answer 仍为纯文本字符串但像 JSON，尝试重新富化
                    if isinstance(final_answer, str) and final_answer.strip().startswith("{"):
                        ans2, items2 = self._extract_final_answer(final_answer)
                        if items2:
                            catalog_items = items2
                            final_answer = ans2
                        elif isinstance(ans2, list):
                            final_answer = ans2
                    # 响应里若已经带 catalog_items（见 _parse_response），并入
                    inline_items_raw = resp.get("catalog_items") or []
                    if inline_items_raw and not catalog_items:
                        try:
                            catalog_items = [
                                CatalogItem(
                                    cataId=str(x.get("cataId", "")),
                                    catalogName=str(x.get("catalogName", "")),
                                    relatedness=float(x.get("relatedness") or 0.0),
                                    organ=str(x.get("organ", "")),
                                )
                                for x in inline_items_raw if isinstance(x, dict)
                            ]
                        except Exception:
                            catalog_items = []
                    final_answer_time = resp["timestamp"]
                    has_final_answer = True

                for chunk in resp.get("chunks", []):
                    chunks.append(
                        ChunkInfo(
                            chunk_id=chunk.get("chunk_id", ""),
                            content=chunk.get("content", ""),
                            score=chunk.get("score", 0.0),
                            metadata=chunk.get("metadata", {}),
                        )
                    )

            if not has_final_answer:
                for step in turn_data["steps"]:
                    resp = step["response"]
                    resp_answer = resp.get("answer") or ""
                    response_type = resp.get("response_type", "")
                    if resp_answer and response_type not in ["intent", "tool_call"]:
                        final_answer = self._extract_content(resp_answer)
                        final_answer_time = resp["timestamp"]
                        break

            if not has_error:
                has_error = "Milvus" in str(turn_data) or "Exception" in str(turn_data)

            is_successful = has_final_answer and not has_error and not has_security_intercept

            # ---- 变更 #2/#3：优先用 AnswerIntegrateNode 日志标记的答案覆盖最终答案 ----
            # 关联方式：以标记本身为锚点，按"最近前置提问"一对一绑定（见 _assign_ai_markers_to_turns）。
            # 规则：命中 [AnswerIntegrateNode] 已回填（目录检索目录回填）→ 取目录清单；
            #       否则命中 'AnswerIntegrateNode节点记录原始的答案' → 取 JSON 的 answer 字段；
            #       都没命中则保留上面从 final_answer 响应提取的结果（兜底）。
            _marker = self._turn_ai_marker.get(i)
            if _marker:
                if _marker["type"] == "backfill":
                    final_answer = _marker["content"]
                    has_final_answer = True
                    if not catalog_items:
                        catalog_items = self._parse_cataids_from_text(_marker["content"])
                elif _marker["type"] == "raw":
                    _ans = self._extract_answer_from_raw_marker(_marker["content"])
                    if _ans:
                        final_answer = _ans
                        has_final_answer = True


            # ---- P1 1.3：cost_time_ms=0 的步骤回填 ----
            # 若有步骤 cost 为 0 但 total_cost_ms（从步骤求和）远小于实际端到端耗时，
            # 将差额均摊或放到最后/内容匹配阶段，避免"0.0s"误导。
            total_cost = float(sum(all_costs))
            try:
                t0 = self._parse_ts(turn_data["question_time"])
                t1 = self._parse_ts(final_answer_time or turn_data["question_time"])
                if t0 and t1 and t1 >= t0:
                    wall_ms = (t1 - t0).total_seconds() * 1000.0
                    if wall_ms > total_cost + 50.0 and zero_cost_step_count > 0:
                        backfill = wall_ms - total_cost
                        # 优先补内容匹配（最可能被后端省略 cost 的阶段）
                        total_cost = wall_ms
            except Exception:
                pass

            additional_metadata = {
                "has_security_intercept": has_security_intercept,
                "started_by_intent_anchor": bool(anchor_flags[i]) if i < len(anchor_flags) else False,
            }

            phase_times = self._analyze_phase_times(turn_data["steps"])

            # 如果回填了差额且 content_matching_time_ms 被标记 0，把差额计入内容匹配
            if (
                zero_cost_step_count > 0
                and phase_times.get("content_matching_time_ms", 0) == 0
                and total_cost > sum(phase_times.get(k, 0) for k in [
                    "intent_recognition_time_ms", "tool_selection_time_ms",
                    "embedding_time_ms", "content_matching_time_ms", "final_answer_time_ms"]) + 50
            ):
                delta = total_cost - sum(
                    phase_times.get(k, 0) for k in [
                        "intent_recognition_time_ms", "tool_selection_time_ms",
                        "embedding_time_ms", "content_matching_time_ms", "final_answer_time_ms"
                    ]
                )
                if delta > 0:
                    phase_times["content_matching_time_ms"] = phase_times.get("content_matching_time_ms", 0) + delta

            phase_times_total = sum(
                phase_times.get(k, 0) for k in [
                    "intent_recognition_time_ms", "tool_selection_time_ms",
                    "embedding_time_ms", "content_matching_time_ms", "final_answer_time_ms"
                ]
            )
            phase_times["total_assigned_ms"] = phase_times_total
            phase_times["total_cost_ms"] = total_cost

            # 如果 catalog_items 为空但 final_answer 是 list（目录/案例），造一份 CatalogItem 方便后续排序
            if not catalog_items and isinstance(final_answer, list):
                for x in final_answer:
                    if isinstance(x, dict) and (x.get("cataId") or x.get("catalogId")):
                        ci = CatalogItem(
                            cataId=str(x.get("cataId") or x.get("catalogId", "")),
                            catalogName=str(x.get("catalogName") or self.catalog_name_map.get(
                                str(x.get("cataId") or x.get("catalogId", "")), "")),
                            relatedness=float(x.get("relatedness") or x.get("catalogRelatedness") or 0.0),
                            organ=str(x.get("organ", "")),
                        )
                        catalog_items.append(ci)

            # 构造 turn 先不带 error_type，随后调用 classify_error_type（依赖 turn 字段）
            turn = LogTurn(
                turn_number=i + 1,
                date=turn_data["question_date"],
                question_time=turn_data["question_time"],
                answer_time=final_answer_time or turn_data["question_time"],
                user_question=turn_data["question"],
                intent_result=intent_result,
                intent_confidence=intent_confidence,
                selected_tool=selected_tool,
                system_answer=self._clean_answer(final_answer) if isinstance(final_answer, str) else final_answer,
                catalog_items=catalog_items,
                total_cost_ms=total_cost,
                total_cost_seconds=round(total_cost / 1000, 2),
                api_call_count=len(turn_data["steps"]),
                step_costs_ms=all_costs,
                zero_cost_step_count=zero_cost_step_count,
                chunks=chunks,
                ws_chunks=turn_data.get("ws_chunks", []),
                request_urls=turn_data["urls"],
                response_statuses=turn_data["statuses"],
                is_successful=is_successful,
                error_message=error_message if not is_successful else "",
                response_types=response_types_list,
                additional_metadata=additional_metadata,
                intent_recognition_time_ms=phase_times["intent_recognition_time_ms"],
                tool_selection_time_ms=phase_times["tool_selection_time_ms"],
                embedding_time_ms=phase_times["embedding_time_ms"],
                content_matching_time_ms=phase_times["content_matching_time_ms"],
                final_answer_time_ms=phase_times["final_answer_time_ms"],
                phase_times=phase_times,
                phase_by_host={k: dict(v) for k, v in phase_by_host.items()},
            )
            turn.error_type = self.classify_error_type(turn)
            self.turns.append(turn)

        # P0 1.1：intent_anchor 模式下，锚点 = 用户真实提问，**完全不做相似度合并**。
        # （相同问题被重复提问也是真实的独立轮次，不应合并。）
        if strategy != "intent_anchor":
            before = len(self.turns)
            self.turns = self._merge_similar_turns(self.turns, self._raw_turns_data)
            print(f"[INFO] {strategy} 模式：合并相似轮次 ({before} → {len(self.turns)})")
        else:
            print(f"[INFO] intent_anchor 模式：跳过相似度合并（保留 {len(self.turns)} 条原始锚点轮次）")

        for i, turn in enumerate(self.turns):
            turn.turn_number = i + 1

        print(f"[INFO] 最终输出 {len(self.turns)} 个有效对话")

        return self.turns

    # ------------------------------------------------------------------
    # 辅助：统一解析 timestamp → datetime / None（P1 3.1 配套）
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_ts(ts: str):
        if not ts:
            return None
        from datetime import datetime
        # YYYY-MM-DDTHH:MM:SS.mmm+TZ
        try:
            if "T" in ts:
                # 去掉 +08:00 / Z 尾，统一当本地时间处理（精确到秒足够）
                head = ts.split("+")[0].split("Z")[0]
                return datetime.fromisoformat(head)
            head = ts
            # "YYYY-MM-DD HH:MM:SS[.ffffff]"
            if "." in head:
                head = head.split(".")[0]
            return datetime.strptime(head[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 辅助：把单步 step 归类到 phase 键（用于 phase_by_host 累积）
    # ------------------------------------------------------------------
    def _classify_step_phase(self, step: Dict, all_steps: List[Dict]) -> str:
        req = step.get("request") or {}
        resp = step.get("response") or {}
        url = req.get("url", "")
        response_type = resp.get("response_type", "") or ""
        if response_type == "intent":
            return "intent_recognition_time_ms"
        if response_type == "empty":
            return "tool_selection_time_ms"
        if response_type == "final_answer":
            return "final_answer_time_ms"
        if "/v1/embeddings" in url:
            return "embedding_time_ms"
        # 无 strong signal：按位置规则
        idx = -1
        for i, s in enumerate(all_steps):
            if s is step:
                idx = i
                break
        n = len(all_steps)
        if n == 1:
            return "final_answer_time_ms"
        if idx == 0:
            return "intent_recognition_time_ms"
        if idx == n - 1:
            return "final_answer_time_ms"
        # URL 是 chat/completions 且 idx==1 → tool_selection，否则 content_matching
        if idx == 1 and "/v1/chat/completions" in url:
            return "tool_selection_time_ms"
        return "content_matching_time_ms"

    def _analyze_phase_times(self, steps: List[Dict]) -> Dict[str, float]:
        """分析各阶段耗时 - 优先基于 response_type 标签分类，位置规则兜底

        分类规则（优先级从高到低）：
          response_type="intent"        → 意图识别
          response_type="empty" 或 ""   → 工具选择（ReAct 函数调用/工具调用时的空文本响应）
          response_type="final_answer"  → 最终回答
          response_type="other"         → 内容匹配/中间推理
          未命中以上标签                → 回退"位置序号"规则：
              - 第1个 /v1/chat/completions: 意图识别
              - 第2个 /v1/chat/completions: 工具选择
              - /v1/embeddings: 向量化检索（所有 embedding 请求全量累加，含 query / doc / chunk embedding）
              - 中间步骤: 内容匹配
              - 最后1个 /v1/chat/completions: 最终回答
        """
        phase_times = {
            "intent_recognition_time_ms": 0.0,
            "tool_selection_time_ms": 0.0,
            "embedding_time_ms": 0.0,
            "content_matching_time_ms": 0.0,
            "final_answer_time_ms": 0.0,
        }

        if not steps:
            return phase_times

        # 收集所有步骤的元信息（保留 response_type 用于智能分类）
        chat_meta = []       # [(chat_pos_in_list, step_idx, cost, response_type)]
        embedding_steps = []
        other_steps = []

        for i, step in enumerate(steps):
            req = step.get("request", {})
            resp = step.get("response", {})
            url = req.get("url", "")
            cost_time = resp.get("cost_time_ms", 0)
            response_type = resp.get("response_type", "") or ""

            if not cost_time:
                continue

            if "/v1/embeddings" in url:
                embedding_steps.append((i, cost_time))
            elif "/v1/chat/completions" in url:
                chat_meta.append((len(chat_meta), i, cost_time, response_type))
            else:
                other_steps.append((i, cost_time))

        # ===== Phase 1: 优先按 response_type 分类 =====
        used_positions = set()
        n_chat = len(chat_meta)

        for pos, step_idx, cost, rt in chat_meta:
            if rt == "intent":
                phase_times["intent_recognition_time_ms"] += cost
                used_positions.add(pos)
            elif rt in ("empty", ""):
                # 'empty' / '' 都是 ReAct 框架下典型的工具选择响应（模型返回 function_call，无文本）
                phase_times["tool_selection_time_ms"] += cost
                used_positions.add(pos)
            elif rt == "final_answer":
                phase_times["final_answer_time_ms"] += cost
                used_positions.add(pos)
            elif rt == "other":
                # 中间推理/内容综合（例如拿到向量结果后进行的思考合成）
                phase_times["content_matching_time_ms"] += cost
                used_positions.add(pos)

        # ===== Phase 2: 位置兜底（response_type 为空或未知的步骤） =====
        for pos, step_idx, cost, rt in chat_meta:
            if pos in used_positions:
                continue

            if n_chat >= 3:
                if pos == 0:
                    phase_times["intent_recognition_time_ms"] += cost
                elif pos == 1:
                    phase_times["tool_selection_time_ms"] += cost
                elif pos == n_chat - 1:
                    phase_times["final_answer_time_ms"] += cost
                else:
                    phase_times["content_matching_time_ms"] += cost
            elif n_chat == 2:
                if pos == 0:
                    phase_times["intent_recognition_time_ms"] += cost
                else:
                    phase_times["final_answer_time_ms"] += cost
            elif n_chat == 1:
                # 单步无法判断，若 response_type 缺失则默认为最终回答
                phase_times["final_answer_time_ms"] += cost

        # ===== 向量化 & 其他步骤 =====
        # 用户最终口径：所有 /v1/embeddings 请求全部归入 embedding_time_ms（向量检索），
        # 不再将 doc/chunk embedding 分流到内容匹配。
        for idx, cost in embedding_steps:
            phase_times["embedding_time_ms"] += cost

        # 非 chat / 非 embedding 的请求（业务接口等），归入内容匹配
        for idx, cost in other_steps:
            phase_times["content_matching_time_ms"] += cost

        total_assigned = sum(phase_times.values())
        phase_times["total_assigned_ms"] = total_assigned

        return phase_times

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度"""
        if not str1 or not str2:
            return 0.0
        return SequenceMatcher(None, str1, str2).ratio()

    # 审查 #6 #7：_extract_key_entities / _has_distinct_entities 是定义未引用的死代码，已删除。
    # 审查 #10：_calculate_time_diff_seconds 已在两个调用点内联实现，避免单用途包装函数跳转。

    def _merge_similar_turns(
        self,
        turns: List[LogTurn],
        raw_data: Optional[List[Dict]] = None,
        force_time_s: Optional[float] = None,
        force_sim: Optional[float] = None,
    ) -> List[LogTurn]:
        """合并相似的对话轮次（以用户提问为分界点，机器改写的算同一个问题）

        force_time_s / force_sim：用于 intent_anchor 策略下严格化合并阈值
        """
        if len(turns) <= 1:
            return turns

        # 第一步：标记每个 turn 是否是用户提问
        is_user_question_list = []
        for turn in turns:
            if self.test_set_questions:
                is_user_q = self._is_user_question(turn.user_question)
            else:
                is_user_q = False
            is_user_question_list.append(is_user_q)

        user_q_count = sum(1 for x in is_user_question_list if x)
        print(f"[INFO] 识别到 {user_q_count} 条用户提问（共 {len(turns)} 条）")

        # 严格模式的阈值
        time_thr = force_time_s if force_time_s is not None else (30 if self.test_set_questions else 60)
        sim_thr = force_sim if force_sim is not None else 0.4

        merged = []
        i = 0

        while i < len(turns):
            current = turns[i]
            current_raw = raw_data[i] if raw_data and i < len(raw_data) else None

            # 如果当前是用户提问，或没有测试集：作为新起点向后合并
            if is_user_question_list[i] or not self.test_set_questions:
                j = i + 1
                while j < len(turns):
                    next_turn = turns[j]
                    next_raw = raw_data[j] if raw_data and j < len(raw_data) else None

                    if is_user_question_list[j]:
                        break

                    # 审查 #10：内联 time_diff 计算（原 _calculate_time_diff_seconds），减少跳转
                    t_cur = self._parse_ts(current.question_time)
                    t_nxt = self._parse_ts(next_turn.question_time)
                    time_diff = abs((t_nxt - t_cur).total_seconds()) if (t_cur and t_nxt) else 999999.0
                    similarity = self._calculate_similarity(
                        current.user_question, next_turn.user_question
                    )

                    if time_diff <= time_thr or similarity >= sim_thr:
                        current = self._merge_two_turns(current, next_turn)
                        current_raw = self._merge_raw_data(current_raw, next_raw)
                        j += 1
                    else:
                        break

                merged.append(current)
                i = j
            else:
                if merged:
                    last = merged[-1]
                    last_raw = raw_data[i - 1] if raw_data and i > 0 else None
                    current_raw = raw_data[i] if raw_data and i < len(raw_data) else None

                    # 审查 #10：内联 time_diff 计算（原 _calculate_time_diff_seconds），减少跳转
                    t_last = self._parse_ts(last.question_time)
                    t_curr = self._parse_ts(current.question_time)
                    time_diff = abs((t_curr - t_last).total_seconds()) if (t_last and t_curr) else 999999.0
                    if time_diff <= (2 * time_thr):
                        merged[-1] = self._merge_two_turns(last, current)
                        if raw_data and i > 0 and i < len(raw_data):
                            raw_data[i - 1] = self._merge_raw_data(last_raw, current_raw)
                        i += 1
                    else:
                        merged.append(current)
                        i += 1
                else:
                    merged.append(current)
                    i += 1

        return merged

    # 审查 #8：_is_machine_rewrite 定义未被引用的死代码，已删除（原仅做 has_final_answer + next_has_intent 的未使用判定）。

    def _merge_raw_data(self, raw1: Dict, raw2: Dict) -> Dict:
        """合并两个raw data"""
        if not raw1:
            return raw2
        if not raw2:
            return raw1
        
        merged = {
            "question": raw1.get("question", ""),
            "question_time": raw1.get("question_time", ""),
            "question_date": raw1.get("question_date", ""),
            "steps": raw1.get("steps", []) + raw2.get("steps", []),
            "urls": raw1.get("urls", []) + raw2.get("urls", []),
            "statuses": raw1.get("statuses", []) + raw2.get("statuses", []),
            "response_types": raw1.get("response_types", []) + raw2.get("response_types", []),
            "has_error": raw1.get("has_error", False) or raw2.get("has_error", False),
            "error_message": raw2.get("error_message", "") or raw1.get("error_message", ""),
        }
        
        return merged

    def _merge_two_turns(self, turn1: LogTurn, turn2: LogTurn) -> LogTurn:
        """合并两个turns，保留第一个的原始问题，合并所有步骤，使用最后一个的答案"""
        merged_phase_times = {}
        phase_keys = ["intent_recognition_time_ms", "tool_selection_time_ms",
                      "embedding_time_ms", "content_matching_time_ms",
                      "final_answer_time_ms"]
        for key in phase_keys:
            merged_phase_times[key] = getattr(turn1, key, 0) + getattr(turn2, key, 0)
        merged_phase_times["total_assigned_ms"] = sum(merged_phase_times.values())
        tms = turn1.total_cost_ms + turn2.total_cost_ms
        merged_phase_times["total_cost_ms"] = tms

        # 合并 phase_by_host：逐 host 逐字段累加
        merged_phase_by_host: Dict[str, Dict[str, float]] = {}
        for src in (getattr(turn1, "phase_by_host", None) or {},
                    getattr(turn2, "phase_by_host", None) or {}):
            for host, agg in src.items():
                if host not in merged_phase_by_host:
                    merged_phase_by_host[host] = {}
                for k, v in (agg or {}).items():
                    try:
                        merged_phase_by_host[host][k] = merged_phase_by_host[host].get(k, 0) + float(v)
                    except (TypeError, ValueError):
                        merged_phase_by_host[host][k] = v

        # catalog_items 去重合并（以 cataId 为键）
        merged_catalog_map: Dict[str, CatalogItem] = {}
        for c in (turn1.catalog_items or []) + (turn2.catalog_items or []):
            if getattr(c, "cataId", None) and c.cataId not in merged_catalog_map:
                merged_catalog_map[c.cataId] = c
            elif not getattr(c, "cataId", None):
                merged_catalog_map[id(c)] = c
        merged_catalogs = list(merged_catalog_map.values())

        ans = turn2.system_answer if turn2.system_answer else turn1.system_answer
        merged = LogTurn(
            turn_number=turn1.turn_number,
            date=turn1.date,
            question_time=turn1.question_time,
            answer_time=turn2.answer_time,
            user_question=turn1.user_question,
            intent_result=turn1.intent_result or turn2.intent_result,
            intent_confidence=turn1.intent_confidence or turn2.intent_confidence,
            selected_tool=turn1.selected_tool or turn2.selected_tool,
            system_answer=ans,
            catalog_items=merged_catalogs,
            total_cost_ms=tms,
            total_cost_seconds=round(tms / 1000, 2),
            api_call_count=turn1.api_call_count + turn2.api_call_count,
            step_costs_ms=turn1.step_costs_ms + turn2.step_costs_ms,
            zero_cost_step_count=(getattr(turn1, "zero_cost_step_count", 0)
                                  + getattr(turn2, "zero_cost_step_count", 0)),
            chunks=turn1.chunks + turn2.chunks,
            request_urls=turn1.request_urls + turn2.request_urls,
            response_statuses=turn1.response_statuses + turn2.response_statuses,
            is_successful=turn2.is_successful or turn1.is_successful,
            error_message=turn2.error_message if turn2.error_message else turn1.error_message,
            response_types=turn1.response_types + turn2.response_types,
            additional_metadata={**turn1.additional_metadata, **turn2.additional_metadata},
            intent_recognition_time_ms=merged_phase_times["intent_recognition_time_ms"],
            tool_selection_time_ms=merged_phase_times["tool_selection_time_ms"],
            embedding_time_ms=merged_phase_times["embedding_time_ms"],
            content_matching_time_ms=merged_phase_times["content_matching_time_ms"],
            final_answer_time_ms=merged_phase_times["final_answer_time_ms"],
            phase_times=merged_phase_times,
            phase_by_host=merged_phase_by_host,
        )
        merged.error_type = turn2.error_type or turn1.error_type
        return merged

    # 审查 #9：get_raw_turns 定义未被引用的对外死 API，已删除；
    # 如外部需要原始轮次数据，可直接使用 LogParser._raw_turns_data 字段或扩展新接口。

    def _classify_response(self, response_data: Dict[str, Any]) -> str:
        """分类响应类型: intent, tool_call, final_answer, security_intercept, error, other"""
        content = response_data.get("answer", "")
        status = response_data.get("status", "")
        
        if status != "200":
            return "error"
        
        if not content:
            return "empty"
        
        # 检查是否是安全拦截 (优先检查，因为可能无法解析为JSON)
        if "安全拦截" in content or "content_filter" in str(response_data.get("finish_reason", "")):
            return "security_intercept"
        
        # 检查是否是工具调用
        if response_data.get("has_tool_calls"):
            return "tool_call"
        
        # 尝试解析 JSON 内容
        try:
            content_json = json.loads(content)
            if isinstance(content_json, dict):
                # 检查是否是最终答案（有 resultType）
                if "resultType" in content_json:
                    return "final_answer"
                
                # 检查是否是意图识别（有 userIntent）
                if "userIntent" in content_json or "thinking" in content_json:
                    return "intent"
        except:
            pass
        
        return "other"

    def _extract_final_answer(
        self,
        content: str,
    ) -> Tuple[Union[str, List[Dict[str, Any]]], List[CatalogItem]]:
        """P0 2.1：从响应内容提取最终答案，并富化目录名；返回 (answer, catalog_items)。

        - resultType=Catalog：优先输出 [{cataId, catalogName, relatedness, organ}] 列表，
          同时用 catalog_name_map 回填缺失的中文名；
        - 其它类型：返回纯文本字符串。
        """
        if not content:
            return "", []

        try:
            content_json = json.loads(content)
            if isinstance(content_json, dict):
                result_type = content_json.get("resultType", "")

                # 案例查询
                if result_type in ("Usecase", "UseCase"):
                    usecases = content_json.get("usecases") or content_json.get("useCases") or []
                    if isinstance(usecases, list) and usecases:
                        rich = []
                        for u in usecases:
                            if not isinstance(u, dict):
                                continue
                            cid = str(u.get("cataId") or u.get("catalogId") or "")
                            name = str(u.get("catalogName") or u.get("catalog_name") or "")
                            # 审查 #14：用 dict.get() 替代 "in + 下标" 组合，避免未来重构时漏写 in 判定而抛 KeyError
                            if not name:
                                name = self.catalog_name_map.get(cid, "") or ""
                            uc_name = str(u.get("caseName") or u.get("usecaseName") or u.get("name") or "")
                            rich.append({
                                "cataId": cid,
                                "catalogName": name,
                                "caseName": uc_name,
                                "useCaseRelatedness": float(u.get("useCaseRelatedness") or u.get("relatedness") or 0.0),
                                "organ": str(u.get("organ") or u.get("applyOrgans") or u.get("apply_organs") or ""),
                            })
                        return rich, []

                # 目录检索：目录清单
                if result_type == "Catalog":
                    catalogs_raw = content_json.get("catalogs", []) or []
                    if isinstance(catalogs_raw, list) and catalogs_raw:
                        items: List[CatalogItem] = []
                        rich_list: List[Dict[str, Any]] = []
                        for c in catalogs_raw:
                            if isinstance(c, dict):
                                cid = str(c.get("cataId") or c.get("catalogId") or c.get("id") or "")
                                name = str(c.get("catalogName") or c.get("catalog_name") or c.get("name") or "")
                            elif isinstance(c, str):
                                cid = c.strip()
                                name = ""
                            else:
                                continue
                            # 审查 #14：用 dict.get() 替代 "in + 下标"，避免漏判导致 KeyError
                            if not name:
                                name = self.catalog_name_map.get(cid, "") or ""
                            rel = 0.0
                            organ = ""
                            extra: Dict[str, Any] = {}
                            if isinstance(c, dict):
                                try:
                                    rel = float(c.get("catalogRelatedness") or c.get("relatedness") or 0.0)
                                except (TypeError, ValueError):
                                    rel = 0.0
                                organ = str(c.get("organ") or c.get("applyOrgans") or "")
                                for k, v in c.items():
                                    if k not in {"cataId", "catalogId", "id", "catalogName",
                                                "catalog_name", "name", "catalogRelatedness",
                                                "relatedness", "organ", "applyOrgans"}:
                                        extra[k] = v
                            it = CatalogItem(cataId=cid, catalogName=name, relatedness=rel,
                                             organ=organ, extra=extra)
                            items.append(it)
                            rich_list.append(it.to_dict())
                        return rich_list, items
                    # 非 list：fallback 到通用字段
                    val = content_json.get("catalogs")
                    if isinstance(val, str):
                        return val, []

                if result_type in ["PolicyInterpret", "OperationGuide"]:
                    ans = content_json.get("answer", "")
                    return (ans if isinstance(ans, str) else (json.dumps(ans, ensure_ascii=False) if ans else "")), []

                # 通用字段兜底
                for key in ["answer", "final_answer", "result", "response"]:
                    if key in content_json:
                        val = content_json[key]
                        if isinstance(val, str) and val:
                            return val, []
                        if isinstance(val, dict):
                            return json.dumps(val, ensure_ascii=False), []
                        if isinstance(val, list) and val:
                            first = val[0]
                            if isinstance(first, dict):
                                # 混合目录数组的兜底
                                return json.dumps(val, ensure_ascii=False), []
                            return json.dumps(val, ensure_ascii=False), []
        except Exception:
            pass

        return content, []

    def _extract_content(self, content: str) -> str:
        """从响应内容中提取有意义的文本"""
        if not content:
            return ""
        
        try:
            content_json = json.loads(content)
            if isinstance(content_json, dict):
                for key in ["answer", "catalogs", "final_answer", "result", "response"]:
                    if key in content_json and isinstance(content_json[key], str):
                        return content_json[key]
        except:
            pass
        
        return content

    def _is_valid_question(self, question: str) -> bool:
        if not question or len(question) < 3:
            return False

        invalid_keywords = [
            "# 角色定义",
            "你是一个严格遵循",
            "# Role",
            "You are a strict",
            "建议使用工具",
            "实现用户意图",
            "Tool:",
            "Action:",
            "Observation:",
            "Thought:",
        ]

        for keyword in invalid_keywords:
            if keyword in question:
                return False

        # 过滤太短的关键词（可能是 embedding 的 input）
        if len(question) < 8 and not any(kw in question for kw in ["？", "?", "什么", "怎么", "如何", "哪里"]):
            return False

        # 过滤太长的内容（可能是工具返回的结果）
        if len(question) > 500:
            return False

        return True

    def _is_noise(self, content: str) -> bool:
        """检测内容是否为噪音（包含大量乱码、特殊字符、非中文内容等）"""
        if not content:
            return True
        
        # 统计中文字符占比
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        total_chars = len(content.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return True
        
        chinese_ratio = chinese_chars / total_chars
        
        # 如果中文字符占比低于30%，可能是噪音
        if chinese_ratio < 0.3:
            return True
        
        # 检测是否包含大量特殊字符（如乱码）
        special_chars = sum(1 for char in content if char in ['×', '★', '♦', '◆', '●', '○', '□', '■', '▲', '▼', '◇', '☆', '✦', '✧'])
        if special_chars > 5:
            return True
        
        # 检测是否有大量数字与特殊符号混杂
        if re.search(r'\d{4,}.[×★◆●■▲▼◇☆♦✦✧]', content):
            return True
        
        return False

    def _should_include(self, question: str) -> bool:
        if not self._is_valid_question(question):
            return False

        # 审查 #2：target_questions 白名单（精确匹配 / 子串都算命中），不匹配则直接排除
        if self.target_questions:
            if not any((q in question) or (q == question) for q in self.target_questions if q):
                return False

        if self.target_keywords:
            for kw in self.target_keywords:
                if kw in question:
                    return True
            return False
        return True

    def _parse_request(self, block: str) -> Dict[str, Any]:
        result = {
            "timestamp": "",
            "date": "",
            "url": "",
            "host": "",
            "user_question": "",
            "tool": "",
            "is_tool_call": False,
            "is_intent_anchor": False,  # P0 1.1：命中"对话意图分析助手"=新一轮
        }

        # P1 3.1：扩展时间戳格式，兼容更多情况（毫秒可选/无TZ/纯空格分隔等）
        # 审查 #5：复用模块级共享 BLOCK 正则（与 parse()/parse_response 同一套，不再维护 3 份）
        for pattern in _TIMESTAMP_PATTERNS_BLOCK:
            m = re.search(pattern, block)
            if m:
                result["timestamp"] = m.group(0)
                result["date"] = m.group(0)[:10]
                break

        m = re.search(r"- url:\s*(\S+)", block)
        if m:
            result["url"] = m.group(1)
            try:
                result["host"] = urlparse(m.group(1)).netloc
            except Exception:
                result["host"] = ""

        # P0 1.1：意图分析锚点（比 messages.user_question 更可靠）
        if _INTENT_ANCHOR_RE.search(block):
            result["is_intent_anchor"] = True

        # 对于 embedding 请求，从 input 中提取用户问题
        if "/v1/embeddings" in result["url"]:
            body = self._extract_json_body(block)
            if body and "input" in body:
                inputs = body["input"]
                if isinstance(inputs, list) and len(inputs) > 0:
                    input_text = inputs[0] if isinstance(inputs[0], str) else str(inputs[0])
                    match = re.search(r'#用户问题\s*\n\s*(.+?)(?:\n|$)', input_text)
                    if match:
                        result["user_question"] = match.group(1).strip()
                    elif self._is_valid_question(input_text):
                        result["user_question"] = input_text.strip()
            return result

        body = self._extract_json_body(block)
        if body:
            
            # 对于 LLM 响应（包含 choices），提取工具调用信息
            if "choices" in body and isinstance(body["choices"], list):
                for choice in body["choices"]:
                    msg = choice.get("message", {})
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        result["is_tool_call"] = True
                        # 提取第一个工具调用的名称和参数
                        for tc in tool_calls:
                            func = tc.get("function", {})
                            if func.get("name"):
                                result["tool"] = func["name"]
                            # 尝试从工具参数中提取用户问题
                            arguments = func.get("arguments", "")
                            if arguments:
                                try:
                                    args_json = json.loads(arguments) if isinstance(arguments, str) else arguments
                                    user_input = args_json.get("userInput", "")
                                    if user_input and not result["user_question"]:
                                        result["user_question"] = user_input
                                except:
                                    # 如果参数是字符串格式，尝试正则提取
                                    match = re.search(r'"userInput"\s*:\s*"([^"]+)"', arguments)
                                    if match and not result["user_question"]:
                                        result["user_question"] = match.group(1)
                            if result["tool"]:
                                break
            
            # 对于 chat 请求，从 messages 提取
            if "messages" in body:
                best_question = result["user_question"]  # 优先使用已提取的问题
                
                for msg in body["messages"]:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        # 优先从 "用户输入：" 格式提取（最可靠）
                        q_match = re.search(r"用户输入[：:]\s*([^;；]+)", content)
                        if q_match:
                            question = q_match.group(1).strip()
                            if self._is_valid_question(question) and not best_question:
                                best_question = question
                        
                        # 如果是最后的 user 消息且内容合理
                        if msg.get("role") == "user":
                            # 排除包含系统指令的复杂 prompt
                            if not any(kw in content for kw in ["# 角色定义", "# 可用工具", "# 对话分析", "#处理要求", "你是一个"]):
                                # 排除参考资料（包含 "参考资料" 或大量乱码/特殊字符）
                                if "参考资料" not in content and not self._is_noise(content):
                                    # 排除 JSON 数组格式
                                    if not content.startswith("[") and not content.startswith("{"):
                                        # 确保内容不是太长的系统提示
                                        if len(content) < 500:
                                            question = content.strip()
                                            if self._is_valid_question(question):
                                                # 使用更简单、更短的问题（通常是更直接的用户输入）
                                                if not best_question or len(question) < len(best_question):
                                                    best_question = question
                
                result["user_question"] = best_question

        return result
    
    def _extract_json_body(self, block: str) -> Optional[Dict]:
        """从日志块中提取完整的 JSON body"""
        try:
            # 找到 body: 后的第一个 {
            body_start = block.find("- body:")
            if body_start == -1:
                return None
            
            json_start = block.find("{", body_start)
            if json_start == -1:
                return None
            
            # 使用括号匹配找到完整的 JSON
            depth = 0
            in_string = False
            escape_next = False
            
            for i in range(json_start, len(block)):
                char = block[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == "\\":
                    escape_next = True
                    continue
                
                if char == '"':
                    in_string = not in_string
                    continue
                
                if in_string:
                    continue
                
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = block[json_start:i+1]
                        return json.loads(json_str)
            
            return None
        except Exception:
            return None

    def _parse_response(self, block: str) -> Dict[str, Any]:
        result = {
            "timestamp": "",
            "date": "",
            "status": "",
            "cost_time_ms": 0,
            "answer": "",
            "final_answer": "",
            "user_intent": "",
            "tool": "",
            "intent_confidence": 0.0,
            "chunks": [],
            "response_type": "",
            "has_tool_calls": False,
            "error_message": "",
        }

        # 支持两种时间戳格式
        # 审查 #5：复用模块级共享 BLOCK 正则（与 parse()/parse_request 同一套，不再维护 3 份）
        for pattern in _TIMESTAMP_PATTERNS_BLOCK:
            m = re.search(pattern, block)
            if m:
                result["timestamp"] = m.group(0)
                result["date"] = m.group(0)[:10]
                break

        m = re.search(r"- status code:\s*(\d+)", block)
        if m:
            result["status"] = m.group(1)

        m = re.search(r"req-cost-time:\s*(\d+)", block)
        if m:
            result["cost_time_ms"] = int(m.group(1))

        # 检查是否有错误
        if result["status"] != "200":
            error_match = re.search(r"error.*?:\s*(.+)", block, re.IGNORECASE)
            if error_match:
                result["error_message"] = error_match.group(1)[:100]

        body = self._extract_json_body(block)
        if body:
            if "choices" in body and len(body["choices"]) > 0:
                choice = body["choices"][0]
                msg = choice.get("message", {})
                answer_content = msg.get("content", "")
                result["answer"] = answer_content
                
                # 检查是否有工具调用
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    result["has_tool_calls"] = True
                
                # 分类响应类型
                finish_reason = choice.get("finish_reason", "")
                
                # 尝试解析 JSON 内容
                if answer_content:
                    try:
                        answer_json = json.loads(answer_content)
                        if isinstance(answer_json, dict):
                            # 提取意图信息
                            if "userIntent" in answer_json:
                                result["user_intent"] = answer_json["userIntent"]
                            if "tool" in answer_json:
                                result["tool"] = answer_json["tool"]
                            if "intentConfidence" in answer_json:
                                try:
                                    result["intent_confidence"] = float(answer_json["intentConfidence"])
                                except (ValueError, TypeError):
                                    pass
                            if "chunks" in answer_json:
                                result["chunks"] = answer_json["chunks"]
                                
                            # 检查是否是最终答案
                            if "resultType" in answer_json:
                                ans, _items = self._extract_final_answer(answer_content)
                                result["final_answer"] = ans
                                result["catalog_items"] = [c.to_dict() for c in _items] if _items else []
                    except:
                        pass
                
                # 确定响应类型
                result["response_type"] = self._classify_response(result)
        else:
            # 没有 body，可能是错误日志
            if "Exception" in block or "ERROR" in block:
                result["response_type"] = "error_log"
                error_match = re.search(r"Exception:\s*(.+)", block)
                if error_match:
                    result["error_message"] = error_match.group(1)[:100]

        return result

    def _clean_answer(self, answer) -> str:
        if not answer:
            return ""
        
        # 处理列表或其他非字符串类型
        if isinstance(answer, list):
            answer = '\n'.join(str(item) for item in answer)
        elif not isinstance(answer, str):
            answer = str(answer)

        cleaned = re.sub(r"^```(?:json)?\s*", "", answer)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            answer_json = json.loads(cleaned)
            if isinstance(answer_json, dict):
                for key in ["answer", "final_answer", "result", "response"]:
                    if key in answer_json and isinstance(answer_json[key], str):
                        return answer_json[key].strip()
        except:
            pass

        return cleaned

    # ------------------------------------------------------------------
    # 变更 #2：AnswerIntegrateNode 答案标记提取与按时间窗关联
    # ------------------------------------------------------------------
    def _extract_ai_markers(self, lines: List[str]) -> List[Dict[str, Any]]:
        """扫描日志，提取两类 AnswerIntegrateNode 答案标记：
           - raw     : 'AnswerIntegrateNode节点记录原始的答案:{...json...}'（通常跨多行）
           - backfill: '[AnswerIntegrateNode] 已回填 N 个目录到 catalogs：[cataId=>名称, ...]'（可能跨多行）
        匹配到标记行后，向后吞并后续非日志行（直到下一条真正以"时间戳+日志级别"开头的日志行），
        拼成一个完整内容块，避免多行 JSON / 长目录清单被单行截断。
        返回按出现顺序的标记列表，每条含 ts(datetime)/type/content。"""
        markers: List[Dict[str, Any]] = []
        raw_re = re.compile(r"AnswerIntegrateNode节点记录原始的答案\s*:?\s*(.*)$")
        back_re = re.compile(r"\[AnswerIntegrateNode\]\s*已回填\s*(.*)$")
        ts_re = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
        # 真正的新日志行：行首为 时间戳 + 空白 + 日志级别。用于区分"答案内容块"与"下一条日志"，
        # 防止答案内容里偶发出现的时间戳被误判为新日志而提前截断。
        new_log_re = re.compile(
            r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+(?:INFO|WARN|ERROR|DEBUG|TRACE)\b"
        )
        n = len(lines)
        i = 0
        while i < n:
            ln = lines[i]
            m = ts_re.search(ln)
            if not m:
                i += 1
                continue
            ts = self._parse_ts(m.group(1))
            line_marker = None
            tail = ""
            if raw_re.search(ln):
                line_marker = "raw"
                tail = raw_re.search(ln).group(1).strip()
            elif back_re.search(ln):
                line_marker = "backfill"
                tail = back_re.search(ln).group(1).strip()
            if line_marker:
                block = [tail] if tail else []
                j = i + 1
                # 向后吞并非日志行，直到下一条真正日志或文件结束
                while j < n:
                    if new_log_re.match(lines[j]):
                        break
                    block.append(lines[j])
                    j += 1
                content = "\n".join(block).strip()
                markers.append({"ts": ts, "type": line_marker, "content": content})
                i = j  # 跳到新日志行，避免重复处理已吞并内容
            else:
                i += 1
        return markers

    def _extract_answer_from_raw_marker(self, content: str) -> str:
        """从 '节点记录原始的答案' 的 JSON 内容中提取 answer 字段。

        优先级：1) 完整 JSON 解析取 answer；2) 正则提取 "answer" 字段值（兼容转义引号）；
        3) 去掉外层花括号与 "answer": 前缀；4) 退回整段文本。"""
        content = content.lstrip(":").strip()
        # 1) 完整 JSON 解析
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and obj.get("answer"):
                return str(obj["answer"])
        except Exception:
            pass
        # 2) 正则提取 "answer" 字段值（兼顾转义引号）
        m = re.search(r'(?s)"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', content)
        if m:
            try:
                return m.group(1).encode("utf-8").decode("unicode_escape")
            except Exception:
                return m.group(1)
        # 3) 去掉外层花括号与 "answer": 前缀
        if content.startswith("{"):
            stripped = content
            stripped = re.sub(r'(?s)^\s*\{\s*"answer"\s*:\s*', "", stripped)
            stripped = re.sub(r'(?s)\s*\}\s*$', "", stripped)
            if stripped:
                return stripped
        return content if content else ""

    def _parse_cataids_from_text(self, text: str) -> List["CatalogItem"]:
        """从 '已回填' 目录清单文本中解析 32 位 cataId（形如 cataId=>名称）。"""
        ids = re.findall(r"[0-9a-fA-F]{32}", text or "")
        items: List["CatalogItem"] = []
        for cid in ids:
            items.append(CatalogItem(cataId=cid, catalogName="", relatedness=0.0, organ=""))
        return items

    def _assign_ai_markers_to_turns(self, turns_data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """以 AnswerIntegrateNode 标记为锚点，做一对一关联（修复错位 + 目录检索回填优先）。

        核心思路：
          - 每个标记归属「提问时间 ≤ 标记时间 的最大者」的那一轮（最近前置提问），避免相邻轮互抢。
          - 同一轮若同时命中 raw 与 backfill，优先取 backfill（目录检索目录回填清单，
            含 cataId=>名称，比 raw 的纯 cataId 更完整）。
          - 未被任何轮次覆盖的标记（如超时/异常轮）自动跳过。

        返回 {turn_index: marker_dict}。"""
        markers = getattr(self, "_ai_markers", None) or []
        if not markers or not turns_data:
            return {}
        # 1) 计算每个轮次的提问时间（升序）
        q_items = []
        for i, td in enumerate(turns_data):
            qt = self._parse_ts(td.get("question_time") or "")
            if qt is not None:
                q_items.append((qt, i))
        q_items.sort(key=lambda x: x[0])

        # 2) 每个标记归属「提问时间 严格小于 标记时间 的最大轮次」（最近前置提问）。
        #    采用「严格小于」优先可消解「答案时间戳 == 下一轮提问时间戳」导致的跨轮错配；
        #    仅当没有任何 q < ts 时，才退化为「≤ ts」的最大轮次。
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for m in markers:
            mts = m.get("ts")
            if mts is None:
                continue
            strict_idx = None  # 严格小于 mts 的最大轮次
            loose_idx = None     # ≤ mts 的最大轮次（兜底）
            for qt, idx in q_items:
                if qt <= mts:
                    loose_idx = idx
                    if qt < mts:
                        strict_idx = idx
                else:
                    break
            best_idx = strict_idx if strict_idx is not None else loose_idx
            if best_idx is None:
                continue
            buckets.setdefault(best_idx, []).append(m)

        # 3) 每轮从命中标记中优先 backfill，否则 earliest raw
        assignment: Dict[int, Dict[str, Any]] = {}
        for idx, ms in buckets.items():
            backfills = [m for m in ms if m["type"] == "backfill"]
            if backfills:
                chosen = min(backfills, key=lambda m: m["ts"])
            else:
                chosen = min(ms, key=lambda m: m["ts"])
            assignment[idx] = chosen
        print(
            f"[INFO] AnswerIntegrateNode 标记一对一关联: {len(assignment)}/{len(markers)} 标记已绑定到轮次"
        )
        return assignment

    # === 变更 #4：WS检索元数据(前端结构) 解析 ===
    _WS_FORMAT_RE = re.compile(r"\[召回片段-(.+?)-WS格式\]")
    _WS_META_RE = re.compile(r"\[WS检索元数据\(前端结构\)\]")
    _WS_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?)")

    def _extract_ws_chunks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """扫描日志，提取 [召回片段-<场景>-WS格式] [<JSON数组>] 行（前端结构检索元数据）。

        每个数组元素形如 {"text":..., "score":..., "embeddingId":..., "source":..., "metadata":{...}}。
        返回按出现顺序的列表，每条含 ts(datetime)/scene/chunks（扁平片段列表）。"""
        out: List[Dict[str, Any]] = []
        for ln in lines:
            m = self._WS_FORMAT_RE.search(ln)
            scene = None
            if m:
                scene = m.group(1).strip()
            else:
                m2 = self._WS_META_RE.search(ln)
                if not m2:
                    continue
            ts_m = self._WS_TS_RE.search(ln)
            ts = self._parse_ts(ts_m.group(1)) if ts_m else None
            # JSON 数组起点：标记 ] 之后的第一个 '['
            marker_end = (m if m else m2).end()
            start = ln.find("[", marker_end)
            if start < 0:
                continue
            candidate = ln[start:]
            end = candidate.rfind("]")
            if end >= 0:
                candidate = candidate[: end + 1]
            try:
                arr = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(arr, list):
                continue
            chunks: List[Dict[str, Any]] = []
            for item in arr:
                if not isinstance(item, dict):
                    continue
                md = item.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}
                if not scene:
                    scene = self._infer_scene(item, md)
                chunks.append(
                    {
                        "scene": scene,
                        "text": item.get("text", ""),
                        "score": item.get("score", 0.0),
                        "embeddingId": item.get("embeddingId"),
                        "source": item.get("source"),
                        "metadata": md,
                    }
                )
            if chunks:
                out.append({"ts": ts, "scene": scene, "chunks": chunks})
        return out

    @staticmethod
    def _infer_scene(item: Dict[str, Any], md: Dict[str, Any]) -> str:
        """[WS检索元数据(前端结构)] 行无显式场景标签，按 metadata 推断。"""
        if "目录标识" in md or "目录编码" in md:
            return "目录检索"
        docnm = md.get("docnm_kwd") or ""
        if "案例查询" in docnm:
            return "案例查询"
        if "政策" in docnm:
            return "政策"
        if "操作指引" in docnm:
            return "操作指引"
        src = item.get("source") or ""
        return src or "检索"

    def _assign_ws_chunks_to_turns(self, turns_data: List[Dict[str, Any]]) -> None:
        """将 WS检索元数据 按时间戳归属到对应轮次（复用最近前置提问逻辑）。

        与 _assign_ai_markers_to_turns 同理：每个检索块归属「提问时间 ≤ 块时间 的最大轮次」，
        严格小于优先。一轮可能命中多个场景/多次检索，全部扁平追加到该轮 ws_chunks。"""
        ws = getattr(self, "_ws_chunks_raw", None) or []
        if not ws or not turns_data:
            return
        q_items = []
        for i, td in enumerate(turns_data):
            qt = self._parse_ts(td.get("question_time") or "")
            if qt is not None:
                q_items.append((qt, i))
        q_items.sort(key=lambda x: x[0])

        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for entry in ws:
            mts = entry.get("ts")
            if mts is None:
                continue
            strict_idx = None
            loose_idx = None
            for qt, idx in q_items:
                if qt <= mts:
                    loose_idx = idx
                    if qt < mts:
                        strict_idx = idx
                else:
                    break
            best_idx = strict_idx if strict_idx is not None else loose_idx
            if best_idx is None:
                continue
            buckets.setdefault(best_idx, []).extend(entry["chunks"])

        for i, chunks in buckets.items():
            seen = set()
            uniq = []
            for c in chunks:
                key = c.get("embeddingId") or c.get("text")
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(c)
            turns_data[i]["ws_chunks"] = uniq
        print(
            f"[INFO] WS检索元数据关联: {len(buckets)}/{len(turns_data)} 轮次含检索块，"
            f"共 {sum(len(v) for v in buckets.values())} 条片段"
        )

    def _markers_in_window(self, q_time_str, next_q_time_str) -> List[Dict[str, Any]]:
        """返回落在 [q_time, next_q_time) 时间窗内的 AnswerIntegrateNode 标记。"""
        markers = getattr(self, "_ai_markers", None) or []
        if not markers:
            return []
        q_ts = self._parse_ts(q_time_str) if q_time_str else None
        n_ts = self._parse_ts(next_q_time_str) if next_q_time_str else None
        res = []
        for m in markers:
            ts = m["ts"]
            if ts is None:
                continue
            if q_ts and ts < q_ts:
                continue
            if n_ts and ts >= n_ts:
                continue
            res.append(m)
        return res

    def export_to_json(self, output_file: str):
        data = [turn.to_dict() for turn in self.turns]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[INFO] 已导出 {len(data)} 个对话轮次到: {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """生成解析后的聚合统计。

        字段单位约定（审查 #13）：
          * 单位 **ms** 字段：avg_total_cost_ms、p50/p90/p99_total_cost_ms、
            phase_*_ms、step_*_ms、top_slow_calls.cost_ms、tool_avg_cost.* 等。
          * 单位 **秒 (s)** 字段：phase_conditional_avg_s、e2e_avg_seconds（通常用于相位条件均值 / 端到端耗时）。
          * 其余：整数（count / n）、字符串、比例（0~1 小数）。
        """
        if not self.turns:
            return {}

        total_cost = sum(t.total_cost_ms for t in self.turns)
        avg_cost = total_cost / len(self.turns)
        max_cost = max(t.total_cost_ms for t in self.turns)
        min_cost = min(t.total_cost_ms for t in self.turns)
        total_api_calls = sum(t.api_call_count for t in self.turns)
        total_zero_cost_steps = sum(getattr(t, "zero_cost_step_count", 0) for t in self.turns)

        successful_turns = sum(1 for t in self.turns if t.is_successful)
        failed_turns = len(self.turns) - successful_turns

        # ---- 工具维度统计：数量、总耗时、成功率（P2 3.5） ----
        tool_counts: Dict[str, int] = defaultdict(int)
        tool_total_cost: Dict[str, float] = defaultdict(float)
        tool_succ_count: Dict[str, int] = defaultdict(int)
        for t in self.turns:
            tool = t.selected_tool or "<未识别>"
            tool_counts[tool] += 1
            tool_total_cost[tool] += t.total_cost_ms
            if t.is_successful:
                tool_succ_count[tool] += 1
        tool_avg_cost = {
            k: round(tool_total_cost[k] / tool_counts[k], 2)
            for k in tool_counts
        }
        tool_success_rate = {
            k: round(tool_succ_count[k] / tool_counts[k] * 100, 2)
            for k in tool_counts
        }

        # ---- 错误维度统计（P2 3.5）：错误类型分布 ----
        error_type_breakdown: Dict[str, int] = defaultdict(int)
        for t in self.turns:
            et = getattr(t, "error_type", "") or ""
            if not et:
                if t.additional_metadata.get("has_security_intercept"):
                    et = "安全拦截"
                elif not t.is_successful:
                    et = "其他"
                else:
                    et = "成功"
            error_type_breakdown[et] += 1

        date_counts = {}
        for t in self.turns:
            date_counts[t.date] = date_counts.get(t.date, 0) + 1

        response_type_counts = {}
        for t in self.turns:
            for rt in t.response_types:
                response_type_counts[rt] = response_type_counts.get(rt, 0) + 1

        # ---- 各阶段的整体累计耗时 + 非 0 turns 计数（用于条件均值） ----
        phase_sum = defaultdict(float)
        phase_nz: Dict[str, int] = defaultdict(int)
        phase_keys = [
            "intent_recognition_time_ms", "tool_selection_time_ms",
            "embedding_time_ms", "content_matching_time_ms", "final_answer_time_ms",
        ]
        for t in self.turns:
            for k in phase_keys:
                v = getattr(t, k, 0) or 0
                phase_sum[k] += v
                if v > 0:
                    phase_nz[k] += 1

        # ---- 分 host 的聚合（P2 1.4） ----
        host_aggregate: Dict[str, Dict[str, float]] = {}
        for t in self.turns:
            for host, agg in (getattr(t, "phase_by_host", None) or {}).items():
                if host not in host_aggregate:
                    host_aggregate[host] = defaultdict(float)
                for k, v in (agg or {}).items():
                    try:
                        host_aggregate[host][k] = host_aggregate[host].get(k, 0) + float(v)
                    except (TypeError, ValueError):
                        host_aggregate[host][k] = v
        host_aggregate_out = {h: dict(v) for h, v in host_aggregate.items()}

        return {
            "total_turns": len(self.turns),
            "successful_turns": successful_turns,
            "failed_turns": failed_turns,
            "success_rate": round(successful_turns / len(self.turns) * 100, 2),
            "total_cost_ms": total_cost,
            "total_cost_seconds": round(total_cost / 1000, 2),
            "avg_cost_ms": round(avg_cost, 2),
            "max_cost_ms": max_cost,
            "min_cost_ms": min_cost,
            "total_api_calls": total_api_calls,
            "avg_api_calls": round(total_api_calls / len(self.turns), 2),
            "zero_cost_step_total": total_zero_cost_steps,
            "tool_distribution": dict(tool_counts),
            # P2 3.5：新增工具维度统计
            "tool_avg_cost_ms": tool_avg_cost,
            "tool_success_rate": tool_success_rate,
            # P2 3.5：错误类型分布
            "error_type_breakdown": dict(error_type_breakdown),
            # 各阶段汇总耗时
            "phase_summary_ms": {k: round(v, 2) for k, v in phase_sum.items()},
            # P2：条件均值（仅该 phase 非 0 的 turns 求算术平均，单位 s）
            # 用于对齐 "timeline HTML" 相位卡片口径。
            "phase_conditional_avg_s": {
                k: round((phase_sum[k] / phase_nz[k]) / 1000.0, 3) if phase_nz.get(k) else 0.0
                for k in phase_sum
            },
            "phase_nonzero_count": {k: phase_nz.get(k, 0) for k in phase_sum},
            # P2 1.4：分 host 汇总
            "host_aggregate": host_aggregate_out,
            "date_distribution": date_counts,
            "response_type_distribution": response_type_counts,
            "turn_boundary": self.turn_boundary,
        }
