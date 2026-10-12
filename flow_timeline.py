#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目录检索流程耗时分析 & 流程图生成器
============================================
用法:
  python flow_timeline.py "D:/path/to/ai-data-search.log"
  python flow_timeline.py "D:/path/to/ai-data-search.log" -o "report.html"
  python flow_timeline.py "D:/path/to/ai-data-search.log" -o "report.html" --open

支持两种日志格式:
  格式A: JSON 行日志，含 "user_question" 字段
  格式B: HTTP 请求日志，用户问题嵌入意图识别 prompt 的 "用户输入：XXX" 中

输出:
  交互式 HTML 报告，含流程图、耗时条形图、汇总统计
"""

import re
import os
import sys
import argparse
from datetime import datetime


# ══════════════════════════════════════════════════════════════
#  日志解析器
# ══════════════════════════════════════════════════════════════

class LogParser:
    """解析 AI 助手日志，按 turn 提取目录检索流程"""

    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.turns = []

    def load(self):
        with open(self.filepath, 'r', encoding='utf-8', errors='replace') as f:
            self.lines = f.readlines()
        print(f"[INFO] 加载日志: {self.filepath}")
        print(f"[INFO] 共 {len(self.lines)} 行")
        return self

    def parse(self):
        """主解析：以意图识别块为锚点，按 turn 分组所有事件"""
        # 步骤1: 找到所有意图识别块的起止行
        intent_blocks = self._find_intent_blocks()

        # 步骤2: 找到所有 LLM 调用（含 model + req-cost-time）
        llm_calls = self._find_llm_calls()

        # 步骤3: 找到 Embedding 调用和目录返回数
        embedding_calls = self._find_embedding_calls()
        catalog_counts = self._find_catalog_counts()

        # 步骤4: 按 turn 分组
        self.turns = self._group_turns(intent_blocks, llm_calls, embedding_calls, catalog_counts)

        smart_count = sum(1 for t in self.turns if t.get('intent_tool') == '目录检索')
        print(f"[INFO] 解析完成: {len(self.turns)} 个 turn，目录检索 {smart_count} 个")
        return self

    # ── 意图识别块定位 ──

    def _find_intent_blocks(self):
        """找到所有意图识别 LLM 调用的起止行和用户问题"""
        blocks = []
        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # 意图识别的 HTTP request 块
            if 'HTTP request:' in line:
                # 确认下一个 HTTP response 之前是否包含 "对话意图分析助手"
                is_intent = False
                for j in range(i, min(i + 40, len(self.lines))):
                    if 'HTTP response:' in self.lines[j]:
                        break
                    if '对话意图分析助手' in self.lines[j]:
                        is_intent = True
                        break

                if is_intent:
                    # 找这个块的 model、用户问题、req-cost-time
                    model = None
                    user_question = None
                    cost_ms = None
                    intent_tool = None
                    ts = self._timestamp_near(i - 1)

                    hit_response = False
                    post_response_scanned = 0
                    for j in range(i, min(i + 100, len(self.lines))):
                        lj = self.lines[j]

                        # model
                        if model is None:
                            m = re.search(r'"model"\s*:\s*"([^"]+)"', lj)
                            if m:
                                model = m.group(1)

                        # 用户问题 — 格式A: user_question
                        if user_question is None:
                            m = re.search(r'"user_question"\s*:\s*"(.+?)"', lj)
                            if m:
                                user_question = m.group(1)

                        # 用户问题 — 格式B: 用户输入：XXX;
                        if user_question is None:
                            m = re.search(r'用户输入：(.+?);', lj)
                            if m and m.group(1).strip() and m.group(1).strip() != 'null':
                                user_question = m.group(1).strip()

                        # req-cost-time (可能在 HTTP response 行之前或之后的 headers 行中)
                        if cost_ms is None:
                            m = re.search(r'req-cost-time:\s*(\d+)', lj)
                            if m:
                                cost_ms = int(m.group(1))

                        # intent tool — 从响应 body 中提取
                        if intent_tool is None and 'HTTP response:' in lj:
                            hit_response = True
                            for k in range(j, min(j + 15, len(self.lines))):
                                mt = re.search(r'"tool"\s*:\s*"([^"]+)"', self.lines[k])
                                if mt:
                                    intent_tool = mt.group(1)
                                    break

                        # HTTP response 后继续扫 10 行读取 req-cost-time（headers 中）
                        if hit_response:
                            post_response_scanned += 1
                            if post_response_scanned > 10:
                                break

                    # 格式B 补充: 也尝试从 "建议使用工具" 行提取
                    if user_question is None:
                        for j in range(i - 20, min(i + 40, len(self.lines))):
                            if j >= 0 and '建议使用工具：' in self.lines[j] and '实现用户意图：' in self.lines[j]:
                                m = re.search(r'实现用户意图：(.+)$', self.lines[j])
                                if m:
                                    user_question = m.group(1).strip()
                                    if user_question.endswith('"'):
                                        user_question = user_question[:-1]
                                break

                    if user_question:
                        blocks.append({
                            'line': i,
                            'model': model,
                            'user_question': user_question,
                            'cost_ms': cost_ms,
                            'intent_tool': intent_tool,
                            'timestamp': ts,
                        })

                # 跳过已处理的 request 块
                for j in range(i, min(i + 100, len(self.lines))):
                    if 'HTTP response:' in self.lines[j]:
                        i = j + 1
                        break
                else:
                    i += 1
                continue

            i += 1

        return blocks

    # ── LLM 调用定位 ──

    def _find_llm_calls(self):
        """找出所有 LLM HTTP 调用及其 req-cost-time"""
        calls = []
        i = 0
        while i < len(self.lines):
            if 'HTTP request:' not in self.lines[i]:
                i += 1
                continue

            req_line = i
            model = None
            cost_ms = None
            ctx_type = 'unknown'
            has_tools_body = False
            ts = self._timestamp_near(i - 1)
            hit_response = False
            post_response_scanned = 0

            # 扫描到 HTTP response + 后续 headers
            # CRITICAL: 遇到下一个 HTTP request: 时立即停止，防止嵌套调用被"吞掉"
            for j in range(i, min(i + 250, len(self.lines))):
                lj = self.lines[j]

                # 遇到下一个 HTTP request（且尚未 hit_response）→ 截断当前窗口
                if j > i and not hit_response and 'HTTP request:' in lj:
                    i = j  # 下一轮从此处开始
                    break

                if model is None:
                    m = re.search(r'"model"\s*:\s*"([^"]+)"', lj)
                    if m:
                        model = m.group(1)

                if cost_ms is None:
                    m = re.search(r'req-cost-time:\s*(\d+)', lj)
                    if m:
                        cost_ms = int(m.group(1))

                # 判断类型（用结构性特征，不依赖中文文本）
                if ctx_type == 'unknown':
                    if '对话意图分析助手' in lj:
                        ctx_type = 'intent'
                    elif '政策解读专家' in lj:
                        ctx_type = 'policy_expert'
                    elif 'resultType' in lj:
                        if 'Catalog' in lj:
                            ctx_type = 'final_catalog'
                        else:
                            ctx_type = 'final_answer'

                # 检测是否带 tools（请求体中有 "tools" 字段 → 工具选择调用）
                if not hit_response and '"tools"' in lj and 'function' in lj:
                    has_tools_body = True

                # Embedding 调用检测（bge-m3 模型 → 跳过）
                if model and 'bge-m3' in model:
                    ctx_type = 'embedding'

                if 'HTTP response:' in lj:
                    hit_response = True

                if hit_response:
                    post_response_scanned += 1
                    if post_response_scanned > 10:
                        i = j + 1
                        break

                if j == min(i + 249, len(self.lines) - 1):
                    i = j + 1
            else:
                i += 1
                continue

            # 回填类型：如果未识别但有 tools 字段 → 工具选择
            if ctx_type == 'unknown' and has_tools_body:
                ctx_type = 'tool_selection'
            elif ctx_type == 'unknown':
                ctx_type = 'final_answer'  # fallback

            calls.append({
                'line': req_line,
                'model': model,
                'cost_ms': cost_ms,
                'type': ctx_type,
                'timestamp': ts,
            })

        return calls

    # ── Embedding & Catalog ──

    def _find_embedding_calls(self):
        """定位 bge-m3 embedding 调用"""
        calls = []
        for i, line in enumerate(self.lines):
            if '"model" : "bge-m3"' in line:
                input_text = ''
                cost_ms = None
                for j in range(i, min(i + 8, len(self.lines))):
                    m = re.search(r'"input"\s*:\s*\[\s*"([^"]*)"', self.lines[j])
                    if m:
                        input_text = m.group(1)
                    m = re.search(r'req-cost-time:\s*(\d+)', self.lines[j])
                    if m:
                        cost_ms = int(m.group(1))
                calls.append({
                    'line': i,
                    'input': input_text,
                    'cost_ms': cost_ms if cost_ms else 0,
                    'timestamp': self._timestamp_near(i),
                })
        return calls

    def _find_catalog_counts(self):
        """统计每次目录检索返回的目录条数"""
        counts = []
        in_block = False
        block_start = 0
        block_count = 0

        for i, line in enumerate(self.lines):
            if 'content2Catalog.查询目录:' in line:
                if not in_block:
                    block_start = i
                    block_count = 1
                    in_block = True
                else:
                    block_count += 1
            elif in_block:
                counts.append({
                    'line': block_start,
                    'count': block_count,
                    'timestamp': self._timestamp_near(block_start),
                })
                in_block = False

        if in_block:
            counts.append({
                'line': block_start,
                'count': block_count,
                'timestamp': self._timestamp_near(block_start),
            })
        return counts

    # ── Turn 分组 ──

    def _group_turns(self, intent_blocks, llm_calls, embedding_calls, catalog_counts):
        """以意图识别块为锚点，将 LLM 调用和事件归入各个 turn"""
        turns = []

        for idx, ib in enumerate(intent_blocks):
            # 确定 turn 的行范围
            start_line = ib['line']
            end_line = (
                intent_blocks[idx + 1]['line']
                if idx + 1 < len(intent_blocks)
                else len(self.lines)
            )

            turn = {
                'question': ib['user_question'],
                'intent_tool': ib['intent_tool'],
                'catalog_count': 0,
                'total_ms': 0,
                'steps': {},
                'question_ts': ib['timestamp'],
            }

            # 意图识别步骤
            if ib['cost_ms']:
                turn['steps']['intent'] = {
                    'model': ib['model'],
                    'cost_ms': ib['cost_ms'],
                    'timestamp': ib['timestamp'],
                }

            # 分配 LLM 调用（用结构性类型，不依赖中文匹配）
            for call in llm_calls:
                if not (start_line < call['line'] < end_line):
                    continue
                if call['type'] == 'intent':
                    continue  # 已处理

                if call['type'] == 'tool_selection':
                    turn['steps']['tool_selection'] = call
                    # 通过 response body 中的 tool 名称推断意图
                    # 如果 intent block 已识别出工具，则用它的值
                    if not turn.get('intent_tool'):
                        turn['intent_tool'] = '目录检索'  # 默认

                elif call['type'] == 'final_catalog':
                    turn['steps']['final_answer'] = call
                    if not turn.get('intent_tool'):
                        turn['intent_tool'] = '目录检索'

                elif call['type'] == 'final_answer':
                    if 'final_answer' not in turn['steps']:
                        turn['steps']['final_answer'] = call

                elif call['type'] == 'policy_expert':
                    turn['steps']['policy_expert'] = call

            # 分配 Embedding
            for emb in embedding_calls:
                if start_line < emb['line'] < end_line:
                    turn['steps']['embedding'] = emb
                    break

            # 分配 Catalog count
            for ct in catalog_counts:
                if start_line < ct['line'] < end_line:
                    turn['catalog_count'] += ct['count']

            # 计算总耗时
            total = 0
            for key in ['intent', 'tool_selection', 'embedding', 'final_answer']:
                step = turn['steps'].get(key)
                if step and step.get('cost_ms'):
                    total += step['cost_ms']
            turn['total_ms'] = total

            turns.append(turn)

        return turns

    # ── 工具方法 ──

    def _timestamp_near(self, line_idx, radius=5):
        for j in range(max(line_idx, 0), max(line_idx - radius, 0), -1):
            m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})', self.lines[j])
            if m:
                return m.group(1)
        return None

    def get_smart_search_turns(self):
        return [t for t in self.turns if t.get('intent_tool') == '目录检索']


# ══════════════════════════════════════════════════════════════
#  HTML 报告生成器
# ══════════════════════════════════════════════════════════════

def _fmt(ms):
    """毫秒 → 可读字符串"""
    if ms is None or ms == 0:
        return '-'
    s = ms / 1000
    if s < 1:
        return f'{ms}ms'
    if s < 10:
        return f'{s:.2f}s'
    return f'{s:.1f}s'


def _pct(val, total):
    if total <= 0:
        return 0
    return min(val / total * 100, 100)


def _speed_badge(ms):
    if ms is None:
        return ''
    if ms >= 30000:
        return '<span class="badge badge-slow">🔴 极慢</span>'
    if ms >= 10000:
        return '<span class="badge badge-warn">🟠 偏慢</span>'
    return ''


def generate_report(turns, log_path):
    """生成完整 HTML 报告"""
    valid = [t for t in turns if t['total_ms'] > 0 and t.get('intent_tool') == '目录检索']
    if not valid:
        valid = [t for t in turns if t['total_ms'] > 0]  # fallback

    if not valid:
        return '''<!DOCTYPE html><html><body style="font-family:sans-serif;text-align:center;
          padding:80px;color:#888"><h2>未找到目录检索流程数据</h2>
          <p>请确认日志中是否包含目录检索（search-catalog）的完整调用记录</p></body></html>'''

    total = len(valid)
    avg_ms = sum(t['total_ms'] for t in valid) / total
    max_ms = max(t['total_ms'] for t in valid)
    min_ms = min(t['total_ms'] for t in valid)
    avg_cat = sum(t['catalog_count'] for t in valid) / total

    log_name = os.path.basename(log_path)

    cards = [_render_card(i + 1, t) for i, t in enumerate(valid)]

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>目录检索流程耗时分析</title>
<style>
:root{{--bg:#f5f6f8;--card:#fff;--text:#1a1a2e;--muted:#6b7280;--border:#e5e7eb;
  --blue:#3b82f6;--green:#10b981;--orange:#f59e0b;--red:#ef4444;--purple:#8b5cf6;--teal:#14b8a6;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--text);padding:28px 20px 60px;line-height:1.6}}
.header{{text-align:center;padding:36px 20px 28px}}
.header h1{{font-size:26px;font-weight:700;margin-bottom:6px}}
.header .meta{{font-size:13px;color:var(--muted)}}
.summary{{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;
  max-width:1000px;margin:0 auto 36px}}
.stat{{background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:18px 28px;text-align:center;min-width:130px}}
.stat .val{{font-size:30px;font-weight:700}}
.stat .lbl{{font-size:12px;color:var(--muted);margin-top:4px}}
.container{{max-width:1000px;margin:0 auto 48px}}

.flow{{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:24px 28px;margin-bottom:22px;transition:box-shadow .2s}}
.flow:hover{{box-shadow:0 3px 16px rgba(0,0,0,0.07)}}
.flow-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px}}
.flow-top h3{{font-size:15px;font-weight:600;max-width:72%;line-height:1.4}}
.flow-top .total-ms{{text-align:right;font-size:13px;color:var(--muted)}}
.flow-top .total-ms b{{display:block;font-size:26px;font-weight:700;color:var(--red)}}
.qbox{{background:#eff6ff;border-left:3px solid var(--blue);padding:10px 15px;
  border-radius:5px;margin-bottom:16px;font-size:14px;line-height:1.5}}

.steps{{display:flex;gap:3px;margin-bottom:14px}}
.step{{flex:1;text-align:center;padding:12px 6px;border-radius:8px;background:#f9fafb}}
.step .n{{width:30px;height:30px;border-radius:50%;display:inline-flex;align-items:center;
  justify-content:center;font-size:14px;font-weight:700;color:#fff;margin-bottom:5px}}
.step .l{{font-size:11px;color:var(--muted);margin-bottom:2px}}
.step .t{{font-size:17px;font-weight:700}}

.c-s1 .n{{background:var(--purple)}}.c-s1 .t{{color:var(--purple)}}
.c-s2 .n{{background:var(--blue)}}.c-s2 .t{{color:var(--blue)}}
.c-s3 .n{{background:var(--teal)}}.c-s3 .t{{color:var(--teal)}}
.c-s4 .n{{background:var(--red)}}.c-s4 .t{{color:var(--red)}}

.bars{{margin-top:6px}}
.bar{{display:flex;align-items:center;gap:10px;margin-bottom:5px}}
.bar .lbl{{width:120px;font-size:12px;color:var(--muted);text-align:right;flex-shrink:0}}
.bar .wrap{{flex:1;height:24px;background:#e5e7eb;border-radius:5px;overflow:hidden}}
.bar .fill{{height:100%;border-radius:5px;display:flex;align-items:center;padding-left:10px;
  font-size:11px;color:#fff;font-weight:600;transition:width .6s cubic-bezier(.4,0,.2,1)}}
.f-s1{{background:var(--purple)}}.f-s2{{background:var(--blue)}}
.f-s3{{background:var(--teal)}}.f-s4{{background:var(--red)}}

.info{{font-size:12px;color:var(--muted);margin-top:12px;padding-top:10px;
  border-top:1px solid var(--border);display:flex;gap:20px;flex-wrap:wrap}}
.badge{{display:inline-block;font-size:11px;padding:2px 9px;border-radius:10px;
  margin-left:8px;font-weight:600}}
.badge-slow{{background:#fde8e8;color:var(--red)}}
.badge-warn{{background:#fff7ed;color:var(--orange)}}

.sec-title{{font-size:19px;font-weight:700;margin:36px 0 16px;padding-bottom:8px;
  border-bottom:2px solid var(--border)}}

@media(max-width:700px){{
  .steps{{flex-direction:column;gap:6px}}
  .bar .lbl{{width:80px;font-size:10px}}
  .summary{{gap:8px}}.stat{{padding:12px 18px;min-width:90px}}
}}
</style>
</head>
<body>
<div class="header">
  <h1>目录检索 · 流程耗时分析</h1>
  <div class="meta">日志: {log_name} &nbsp;|&nbsp; 生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
</div>

<div class="summary">
  <div class="stat"><div class="val" style="color:var(--purple)">{total}</div><div class="lbl">目录检索 Turn</div></div>
  <div class="stat"><div class="val" style="color:var(--red)">{_fmt(int(avg_ms))}</div><div class="lbl">平均整体耗时</div></div>
  <div class="stat"><div class="val" style="color:var(--orange)">{_fmt(max_ms)}</div><div class="lbl">最长耗时</div></div>
  <div class="stat"><div class="val" style="color:var(--blue)">{_fmt(min_ms)}</div><div class="lbl">最短耗时</div></div>
  <div class="stat"><div class="val" style="color:var(--teal)">{avg_cat:.1f}</div><div class="lbl">平均目录数</div></div>
</div>

<div class="container">
  <div class="sec-title">  流程详情 (共 {total} 个)</div>
  {''.join(cards)}
</div>

<script>
document.addEventListener('DOMContentLoaded',function(){{
  document.querySelectorAll('.fill').forEach(function(f){{
    var w=f.style.width;f.style.width='0';
    requestAnimationFrame(function(){{requestAnimationFrame(function(){{f.style.width=w}})}});
  }});
}});
</script>
</body>
</html>'''


def _render_card(idx, turn):
    q = turn['question'][:100]
    total = turn['total_ms']
    steps = turn['steps']

    def sc(key):
        s = steps.get(key)
        if s is None:
            return 0
        if isinstance(s, dict):
            return s.get('cost_ms') or 0
        if isinstance(s, (int, float)):
            return int(s)
        return 0

    i_ms = sc('intent')
    t_ms = sc('tool_selection')
    e_ms = sc('embedding')
    f_ms = sc('final_answer')

    # 步骤方块
    nums = '①②③④'
    step_divs = []
    for css, label, ms in [
        ('c-s1', '意图识别', i_ms),
        ('c-s2', '工具选择', t_ms),
        ('c-s3', '检索定位', e_ms),
        ('c-s4', '最终回答', f_ms),
    ]:
        step_divs.append(f'''<div class="step {css}">
          <div class="n">{nums[len(step_divs)]}</div>
          <div class="l">{label}</div>
          <div class="t">{_fmt(ms)}</div></div>''')

    # 条形图
    bars = []
    for css, label, ms in [
        ('f-s1', '① 意图识别 (判定工具类型)', i_ms),
        ('f-s2', '② 工具选择 (ReAct选工具)', t_ms),
        ('f-s3', '③ Embedding + 向量检索', e_ms),
        ('f-s4', '④ 最终回答 (catalogs格式化)', f_ms),
    ]:
        if ms > 0:
            pct = _pct(ms, total)
            bars.append(f'''<div class="bar"><div class="lbl">{label}</div>
              <div class="wrap"><div class="fill {css}" style="width:{pct:.1f}%">{_fmt(ms)}</div></div></div>''')

    # 模型信息
    i_model = steps.get('intent', {}).get('model', '-') if isinstance(steps.get('intent'), dict) else '-'
    f_model = steps.get('final_answer', {}).get('model', '-') if isinstance(steps.get('final_answer'), dict) else '-'

    info = f'意图模型: {i_model} &nbsp;|&nbsp; 回答模型: {f_model} &nbsp;|&nbsp; 目录: {turn["catalog_count"]}条 &nbsp;|&nbsp; 整体耗时: {_fmt(total)}'

    return f'''<div class="flow">
  <div class="flow-top">
    <h3>#{idx} {turn["question"][:60]}{"…" if len(turn["question"])>60 else ""}{_speed_badge(total)}</h3>
    <div class="total-ms">整体耗时<b>{_fmt(total)}</b></div>
  </div>
  <div class="qbox"><strong>问题：</strong>{turn["question"]}</div>
  <div class="steps">{''.join(step_divs)}</div>
  <div class="bars">{''.join(bars)}</div>
  <div class="info"><span>{info}</span></div>
</div>'''


# ══════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='目录检索流程耗时分析 & 流程图生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例:\n  python flow_timeline.py log.log\n  python flow_timeline.py log.log -o r.html --open')
    ap.add_argument('logfile', help='日志文件路径')
    ap.add_argument('-o', '--output', help='输出 HTML 路径 (默认自动生成到 output/ 目录)')
    ap.add_argument('--open', action='store_true', help='生成后自动在浏览器打开')
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        print(f'[ERROR] 文件不存在: {args.logfile}')
        sys.exit(1)

    # 解析
    parser = LogParser(args.logfile)
    parser.load().parse()

    turns = parser.get_smart_search_turns()
    if not turns:
        print('[WARN] 日志中未找到目录检索流程，尝试展示所有有效 turn…')
        turns = [t for t in parser.turns if t['total_ms'] > 0]

    # 生成 HTML
    html = generate_report(turns, args.logfile)

    # 确定输出路径
    if args.output:
        out = args.output
    else:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = os.path.splitext(os.path.basename(args.logfile))[0]
        out = os.path.join(out_dir, f'flow_timeline_{stem}_{ts}.html')

    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f'\n[OK] 报告已生成: {out}')
    print(f'[OK] 共 {len(turns)} 个目录检索流程 / {len(parser.turns)} 个总 turn')

    if args.open:
        import webbrowser
        webbrowser.open(f'file:///{out}')

    return out


if __name__ == '__main__':
    main()
