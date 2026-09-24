# -*- coding: utf-8 -*-
"""㉑ 跑 10 道题评测，出「逐题记录」（召回块 / 对错 / 错在哪）。

判分方式（两层，都记下来，不一致的单列）：
  · 自动：key_facts 命中率 + gold 块是否进 top-5/top-10（检索层 vs 生成层分开看）
  · 人工：verdict（0 错 / 1 部分对 / 2 对）与 error_type（E1–E7，见下），
         跑完后由我看答案逐题判并写进 CSV（脚本先给自动分）

错误类型编码（就是"错在哪"的结构化答案）：
  E1 检索未召回（gold 不在 top-10：检索的锅）
  E2 召回了但答错（生成的锅）      E3 数字/单位错（元 vs 亿元）
  E4 跨公司错位（把 A 的数安到 B 头上）  E5 幻觉（任何块里都找不到）
  E6 假性拒答（原文里有却说没有）  E7 该拒答却硬答

运行（nlp 环境）：
    python scripts/21_run_eval_a.py [--limit 3]
产出：
    outputs/eval/results_a.csv   outputs/eval/results_a.md
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_lib import Retriever, answer  # noqa: E402

from config import EVAL_DIR, OUT_DIR  # noqa: E402

KFF = EVAL_DIR / "key_facts.json"          # 参考答案要点（从财报原文里摘，跑之前定）


def norm(s: str) -> str:
    return re.sub(r"[\s,，]", "", s)


def fact_hit(facts: list[dict], text: str) -> tuple[int, list[str]]:
    """key_facts 命中情况：每个 fact 给出若干等价写法，命中任一即算中。"""
    t = norm(text)
    hit, miss = 0, []
    for f in facts:
        ok = any(norm(v) in t for v in f["any"])
        if ok:
            hit += 1
        else:
            miss.append(f["name"])
    return hit, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = json.loads((EVAL_DIR / "questions_a.json").read_text(encoding="utf-8"))
    kfs = json.loads(KFF.read_text(encoding="utf-8")) if KFF.exists() else {}
    if not KFF.exists():
        print("⚠️  没有 eval/key_facts.json，将只记录召回与角标，不做事实命中判定")
    if args.limit:
        qs = qs[:args.limit]

    r = Retriever("a")
    rows = []
    for q in qs:
        qid = q["id"]
        print(f"\n=== {qid} {q['type']} ===")
        res = answer(q["question"], r, top=8)
        blocks = res["blocks"]
        joined = " ".join(b["text"] for b in blocks)
        facts = (kfs.get(qid) or {}).get("facts", [])
        hit, miss = fact_hit(facts, joined) if facts else (None, [])
        gold_pages = (kfs.get(qid) or {}).get("gold", [])
        # gold 是否进 top-5 / top-10（按"公司+页码"匹配）
        def gold_rank(blocks_, pages):
            for i, b in enumerate(blocks_, 1):
                for g in pages:
                    if b["company"] == g["company"] and str(b["printed_page"]) == str(g["page"]):
                        return i
            return None
        rank = gold_rank(blocks, gold_pages) if gold_pages else None

        print(f"  召回 {res['n_blocks']} 块（模式 {res['mode']}）｜gold 排名 {rank}")
        print(f"  数字核对：{res['number_check']['unsupported'] or '全部可在原文找到'}")
        print(f"  答案：{res['answer'][:160].replace(chr(10),' ')}…")

        rows.append({
            "id": qid, "type": q["type"], "question": q["question"],
            "mode": res["mode"], "retrieved_ids": ";".join(b["chunk_id"] for b in blocks),
            "gold_rank": rank if rank is not None else "",
            "gold_in_top5": "1" if (rank and rank <= 5) else ("0" if gold_pages else ""),
            "gold_in_top10": "1" if (rank and rank <= 10) else ("0" if gold_pages else ""),
            "fact_hit": f"{hit}/{len(facts)}" if facts else "",
            "fact_miss": ";".join(miss),
            "citations": len(res["citations"]),
            "invalid_citations": len(res["invalid_citations"]),
            "number_check": "ok" if res["number_check"]["ok"] else f"可疑{len(res['number_check']['unsupported'])}处",
            "unsupported": ";".join(res["number_check"]["unsupported"][:6]),
            "answer": res["answer"],
            "verdict": "", "error_type": "", "note": "",     # 人工填
        })

    out_csv = OUT_DIR / "eval" / "results_a.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {out_csv}")

    print("\n=== 汇总 ===")
    n = len(rows)
    got = [r_ for r_ in rows if r_["gold_in_top10"] != ""]
    print(f"  题数 {n}｜gold 进 top10 比例 {sum(1 for x in got if x['gold_in_top10']=='1')}/{len(got)}")
    print(f"  数字核对通过 {sum(1 for x in rows if x['number_check']=='ok')}/{n}")
    for x in rows:
        print(f"    {x['id']:4s} gold#{x['gold_rank'] or '-':>3} facts {x['fact_hit'] or '-':>5s} "
              f"数字 {x['number_check']:8s} {x['type']}")


if __name__ == "__main__":
    main()
