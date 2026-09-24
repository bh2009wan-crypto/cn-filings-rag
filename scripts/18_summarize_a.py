# -*- coding: utf-8 -*-
"""⑱ 汇总作业 A 的评测结果（把数字算准，结论由人来写）。

运行（crawler 环境）：
    python scripts/18_summarize_a.py
产出：屏幕（按题型分组的命中率/错误分布，供写 docs/结论A_一页.md 时引用）
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVAL_DIR, OUT_DIR  # noqa: E402

CSV_PATH = OUT_DIR / "eval" / "results_a.csv"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"❌ 先跑 scripts/21_run_eval_a.py（缺 {CSV_PATH}）")
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    scored = [r for r in rows if not r["type"].startswith("附加")]
    frozen = (EVAL_DIR / "questions_a.sha256").read_text(encoding="utf-8").strip()
    print(f"题目哈希：{frozen}\n计入评测 {len(scored)} 题，附加 {len(rows)-len(scored)} 题\n")

    def rate(rs, key):
        vals = [r[key] for r in rs if r[key] != ""]
        return f"{sum(1 for v in vals if v == '1')}/{len(vals)}" if vals else "—"

    print("=== 按题型 ===")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    print(f"{'题型':12s} {'题数':4s} {'gold进top10':12s} {'数字核对通过':12s} {'事实命中率(均)':14s}")
    for t, rs in by_type.items():
        facts = [r["fact_hit"] for r in rs if r["fact_hit"]]
        if facts:
            hits = [int(f.split("/")[0]) / max(1, int(f.split("/")[1])) for f in facts]
            fh = f"{sum(hits)/len(hits):.0%}"
        else:
            fh = "—"
        ok = sum(1 for r in rs if r["number_check"] == "ok")
        print(f"{t:12s} {len(rs):>4d} {rate(rs, 'gold_in_top10'):12s} "
              f"{f'{ok}/{len(rs)}':12s} {fh:14s}")

    print("\n=== 检索模式分布 ===")
    print("  ", dict(Counter(r["mode"] for r in rows)))
    print("\n=== 数字核对失败的题 ===")
    for r in rows:
        if r["number_check"] != "ok":
            print(f"  {r['id']}: {r['number_check']} → {r['unsupported']}")
    print("\n=== 角标问题 ===")
    for r in rows:
        if r["invalid_citations"] != "0":
            print(f"  {r['id']}: 非法角标 {r['invalid_citations']} 个")
    print("\n=== 逐题速览 ===")
    for r in rows:
        print(f"  {r['id']:4s} {r['type']:10s} gold#{r['gold_rank'] or '-':>3} "
              f"facts {r['fact_hit'] or '-':>5s} 数字 {r['number_check']:8s} "
              f"答案 {r['answer'][:60].replace(chr(10),' ')}…")

    human = [r for r in rows if r.get("verdict")]
    if human:
        print("\n=== 人工判分（若有） ===")
        print("  verdict:", dict(Counter(r["verdict"] for r in human)))
        print("  error_type:", dict(Counter(r["error_type"] for r in human if r["error_type"])))


if __name__ == "__main__":
    main()
