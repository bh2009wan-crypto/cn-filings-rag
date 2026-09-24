# -*- coding: utf-8 -*-
"""⑧ 问答页面（Flask，跑在 nlp 环境——query 向量要在这里现场算）。

页面回答作业 A 的两条硬要求：
  「能提问」   → 输入框 + 示例题按钮（点一下填入，方便录屏/截图）
  「答案带出处」→ 答案里的 [n] 角标 + 下方证据卡片（公司·章节·页码·表题 + 原文 + BM25/向量名次）
并把「数字核对」显示在页面上——把课件讲的"数字纪律"变成看得见的东西。

运行（nlp 环境）：
    python scripts/08_app.py --set A --port 7860
    浏览器打开 http://127.0.0.1:7860
接口：GET /api/ask?q=…（与页面同一条链，评测脚本直接调它）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, render_template_string, request  # noqa: E402

from ask_lib import Retriever, answer  # noqa: E402

from config import EVAL_DIR, FINAL_TOP  # noqa: E402

app = Flask(__name__)
STATE: dict = {"retr": None, "last": None, "examples": []}

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>白酒 11 家 2026 半年报 · 问答知识库</title>
<style>
 body{font-family:-apple-system,"PingFang SC",sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1a1a1a}
 h1{font-size:20px} h3{font-size:15px;margin:18px 0 8px} .muted{color:#666;font-size:13px}
 textarea{width:100%;height:64px;font-size:15px;padding:8px;box-sizing:border-box;border:1px solid #ccc;border-radius:6px}
 button{font-size:13px;padding:5px 10px;margin:3px 3px 3px 0;cursor:pointer;border:1px solid #ccc;border-radius:6px;background:#fafafa}
 button.go{background:#2a5580;color:#fff;border-color:#2a5580;padding:8px 22px;font-size:15px}
 .card{border:1px solid #e2e2e2;border-left:4px solid #2a5580;padding:10px 12px;margin:10px 0;border-radius:4px;background:#fbfbfb}
 .ans{background:#f5f8fb;border:1px solid #cfe0ee;padding:12px;border-radius:6px;white-space:pre-wrap;line-height:1.7}
 .src{font-size:12px;color:#444;margin-bottom:6px} .q{font-family:monospace;font-size:11px;color:#777}
 .ok{color:#0a7a3d} .bad{color:#b3261e;font-weight:600}
 .tag{display:inline-block;background:#eef3f8;border-radius:4px;padding:1px 6px;font-size:12px;margin-right:6px;color:#2a5580}
 pre{white-space:pre-wrap;font-family:inherit;font-size:13px;color:#333;margin:0}
</style></head><body>
<h1>白酒 11 家 2026 年半年报 · 问答知识库</h1>
<div class="muted">向量（Qwen3-Embedding-0.6B）+ BM25 混合检索 → deepseek-flash 作答；答案里的 [n] 对应下方证据卡片。</div>
<form method="post" action="/ask">
  <p><textarea name="q" placeholder="问点什么，例如：2026 年上半年贵州茅台的营业收入是多少？">{{ q }}</textarea></p>
  <button class="go" type="submit">提问</button>
  <span class="muted">示例题：</span>
  {% for e in examples %}<button type="button" onclick="document.querySelector('textarea').value={{ e|tojson }}">{{ loop.index }}</button>{% endfor %}
</form>
{% if result %}
  <h3>答案</h3>
  <div class="ans">{{ result.answer }}</div>
  <div class="muted" style="margin-top:6px">
    <span class="tag">检索模式 {{ result.mode }}</span>
    <span class="tag">召回 {{ result.n_blocks }} 块</span>
    <span class="tag">角标 {{ result.citations|length }} 处{% if result.invalid_citations %}（非法 {{ result.invalid_citations }}）{% endif %}</span>
    <span class="tag {% if result.number_check.ok %}ok{% else %}bad{% endif %}">
      数字核对 {% if result.number_check.ok %}通过（{{ result.number_check.checked }} 个数字均可在原文找到）{% else %}可疑：{{ result.number_check.unsupported }}{% endif %}</span>
  </div>
  <h3>证据卡片</h3>
  {% for b in result.blocks %}
  <div class="card">
    <div class="src"><b>[{{ loop.index }}]</b>
      <span class="tag">{{ b.company }}</span><span class="tag">{{ b.section }}</span>
      <span class="tag">第 {{ b.printed_page }} 页</span>
      {% if b.table_title %}<span class="tag">表：{{ b.table_title }}</span>{% endif %}
      <span class="q">{{ b.chunk_id }}　BM25#{{ b.bm25_rank }}　向量#{{ b.vec_rank }}</span>
    </div>
    <pre>{{ b.text[:400] }}{% if b.text|length > 400 %}…{% endif %}</pre>
  </div>
  {% endfor %}
{% endif %}
</body></html>"""


def render(q: str = ""):
    return render_template_string(PAGE, q=q, examples=STATE["examples"], result=STATE["last"])


@app.get("/")
def index():
    return render(q="")


@app.post("/ask")
def do_ask():
    q = (request.form.get("q") or "").strip()
    if q:
        STATE["last"] = answer(q, STATE["retr"], top=FINAL_TOP)
    return render(q)


@app.get("/api/ask")
def api_ask():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"error": "缺少 q 参数"}), 400
    return jsonify(answer(q, STATE["retr"], top=FINAL_TOP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="A")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("→ 加载索引与模型 …")
    STATE["retr"] = Retriever(args.set.lower())
    qfile = EVAL_DIR / "questions_a.json"
    if qfile.exists():
        STATE["examples"] = [x["question"] for x in json.loads(qfile.read_text(encoding="utf-8"))]
    print(f"✅ 就绪：{len(STATE['retr'].docs)} 块，示例题 {len(STATE['examples'])} 道")
    print(f"→ 打开 http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
