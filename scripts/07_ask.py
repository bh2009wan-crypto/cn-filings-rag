# -*- coding: utf-8 -*-
"""⑦ 命令行的「问一句」入口（核心逻辑在 scripts/ask_lib.py，页面与评测共用）。

运行（nlp 环境）：
    python scripts/07_ask.py "贵州茅台2026年上半年营业收入是多少？"
    python scripts/07_ask.py --mode fanout "11 家里 2026 上半年营业收入最高的是哪家？"
产出：屏幕（召回块 + 答案 + 角标校验 + 数字核对）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_lib import Retriever, answer  # noqa: E402

from config import FINAL_TOP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--set", default="A")
    ap.add_argument("--mode", default="auto", choices=["auto", "global", "multi", "fanout"])
    ap.add_argument("--top", type=int, default=FINAL_TOP)
    args = ap.parse_args()

    q = " ".join(args.question)
    r = Retriever(args.set.lower())
    res = answer(q, r, mode=args.mode, top=args.top)

    print(f"\n[检索模式 {res['mode']}] 召回 {res['n_blocks']} 块")
    for i, b in enumerate(res["blocks"], 1):
        print(f"  [{i}] {b['chunk_id']:34s} {b['company']:5s} {b['section'][:14]:14s} "
              f"p{b['printed_page']} BM25#{b['bm25_rank']} 向量#{b['vec_rank']}")
    print(f"\n【答案】\n{res['answer']}")
    print(f"\n【角标】{len(res['citations'])} 处，非法：{res['invalid_citations'] or '无'}")
    nc = res["number_check"]
    print(f"【数字核对】{nc['checked']} 个数字；未在原文找到：{nc['unsupported'] or '无'}")


if __name__ == "__main__":
    main()
