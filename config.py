import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List


def _get_env(key: str, default: str = "") -> str:
    """安全获取环境变量，支持 .env 文件"""
    value = os.environ.get(key, default)
    return value


def _get_env_int(key: str, default: int = 0) -> int:
    """安全获取整数环境变量"""
    value = os.environ.get(key, str(default))
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _get_env_float(key: str, default: float = 0.0) -> float:
    """安全获取浮点数环境变量"""
    value = os.environ.get(key, str(default))
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool = True) -> bool:
    """安全获取布尔环境变量，空/未设置=default；1/true/yes/是 为真；其余为假"""
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "是", "真"}


@dataclass
class JudgeModelConfig:
    api_base: str = field(default_factory=lambda: _get_env("JUDGE_API_BASE", "https://api.minimaxi.com/v1"))
    api_key: str = field(default_factory=lambda: _get_env("JUDGE_API_KEY", ""))
    model: str = field(default_factory=lambda: _get_env("JUDGE_MODEL", "MiniMax-M3"))
    timeout: int = field(default_factory=lambda: _get_env_int("JUDGE_TIMEOUT", 120))


@dataclass
class EvalConfig:
    dimensions: List[str] = field(default_factory=lambda: [
        "accuracy", "completeness", "safety", "readability", "latency"
    ])
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": _get_env_float("WEIGHT_ACCURACY", 0.30),
        "completeness": _get_env_float("WEIGHT_COMPLETENESS", 0.25),
        "safety": _get_env_float("WEIGHT_SAFETY", 0.10),
        "readability": _get_env_float("WEIGHT_READABILITY", 0.15),
        "latency": _get_env_float("WEIGHT_LATENCY", 0.20),
    })
    
    min_pass_score: float = field(default_factory=lambda: _get_env_float("MIN_PASS_SCORE", 3.0))
    latency_thresholds: Dict[int, int] = field(default_factory=lambda: {
        5: 5000,
        4: 10000,
        3: 20000,
        2: 30000,
        1: 9999999
    })
    parallel_workers: int = field(default_factory=lambda: _get_env_int("EVAL_WORKERS", 4))


@dataclass
class LogParserConfig:
    log_file: str = field(default_factory=lambda: _get_env("LOG_FILE", ""))
    target_dates: Optional[set] = None
    target_keywords: Optional[List[str]] = None
    # 测评批跑时注入的时间窗口/边界策略（对应 README 审查 #12 新增环境变量）
    target_time_after: Optional[str] = field(default_factory=lambda: _get_env("LOG_TIME_AFTER", "") or None)
    target_time_before: Optional[str] = field(default_factory=lambda: _get_env("LOG_TIME_BEFORE", "") or None)
    turn_boundary: str = field(default_factory=lambda: _get_env("TURN_BOUNDARY", "intent_anchor") or "intent_anchor")
    target_questions: Optional[List[str]] = field(
        default_factory=lambda: [q.strip() for q in _get_env("TARGET_QUESTIONS", "").split(",") if q.strip()] or None
    )
    parse_catalog_metadata: bool = field(
        default_factory=lambda: _get_env_bool("PARSE_CATALOG_METADATA", True)
    )


@dataclass
class TestSetConfig:
    test_set_file: str = field(default_factory=lambda: _get_env("TEST_SET_FILE", ""))
    mapping_file: Optional[str] = field(default_factory=lambda: os.environ.get("MAPPING_FILE", None) or None)
    excel_question_col: int = field(default_factory=lambda: _get_env_int("EXCEL_QUESTION_COL", 3))
    excel_answer_col: int = field(default_factory=lambda: _get_env_int("EXCEL_ANSWER_COL", 4))
    excel_id_col: int = field(default_factory=lambda: _get_env_int("EXCEL_ID_COL", 0))
    excel_scenario_col: int = field(default_factory=lambda: _get_env_int("EXCEL_SCENARIO_COL", 1))
    excel_is_negative_col: int = field(default_factory=lambda: _get_env_int("EXCEL_IS_NEGATIVE_COL", 6))
    excel_header_row: int = field(default_factory=lambda: _get_env_int("EXCEL_HEADER_ROW", 0))


@dataclass
class OutputConfig:
    output_dir: str = field(default_factory=lambda: _get_env("OUTPUT_DIR", "./output"))


@dataclass
class AppConfig:
    judge: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log_parser: LogParserConfig = field(default_factory=LogParserConfig)
    test_set: TestSetConfig = field(default_factory=TestSetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def parse_dates(dates_str: str) -> Optional[set]:
    if not dates_str:
        return None
    dates = set()
    for d in dates_str.split(","):
        d = d.strip()
        if d:
            dates.add(d)
    return dates
