# -*- coding: utf-8 -*-
"""⑯ 给作业 B 的「RAG 对照」准备上下文：用同一套检索（BM25 + 向量）在 B 的语料上
（茅台 2021–2026 共 11 份报告）给每道测试题取回原文，导出 rag_context.json 供 Colab 用。

为什么用 B 的语料而不是只用在 A 建的 11 家 2026H1 索引：
  B 的测试题跨 2021–2026 各报告期，只拿 2026H1 的索引去找，绝大多数题会"检索不到"，
  那样格5 比的就不是"知识库 vs 微调"，而是"库里有没有"。这里让两侧看到同样 11 份报告，
  对比才公平（README 里也写明这一点）。

运行（nlp 环境）：
    python scripts/16_dump_rag_for_b.py [--top 3]
产出：
    data/sft/rag_context.json   {问题: "拼好的检索原文（top-k）"}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_lib import Retriever  # noqa: E402

from config import SFT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    test = [json.loads(l) for l in (SFT_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    para_file = SFT_DIR / "test_para.jsonl"
    if para_file.exists():
        test += [json.loads(l) for l in para_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  测试题 {len(test)} 道（含换问法）")

    r = Retriever("b")
    out, n_empty = {}, 0
    for i, d in enumerate(test, 1):
        blocks, _ = r.retrieve(d["问题"], mode="global", top=args.top)
        if not blocks:
            n_empty += 1
            continue
        parts = []
        for b in blocks:
            loc = f"{b['company']}{b['report_period']}{b['report_type']}·{b['section']}·第{b['printed_page']}页"
            parts.append(f"【{loc}】\n{b['text']}")
        out[d["问题"]] = "\n\n".join(parts)
        if i % 50 == 0 or i == len(test):
            print(f"    {i}/{len(test)}")

    path = SFT_DIR / "rag_context.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 写入 {path}（{len(out)} 条，{n_empty} 道没检索到）")


if __name__ == "__main__":
    main()
