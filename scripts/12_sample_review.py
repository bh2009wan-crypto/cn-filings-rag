# -*- coding: utf-8 -*-
"""⑫ 人工抽检样本：固定种子分层抽 40 条，预填全部上下文，让人不用翻 PDF 就能判。

作业 B 第②步要求「人工抽检，删掉编造的」——这份记录就是那一步的交付物：
带抽样种子、带样本、带编造率，不是一句"我检查过了"。

抽样规则（写进记录里，可复现）：
  · random.Random(20260924)
  · 按报告期分层，每份报告至少 3 条
  · 额外强制包含 5 条"边缘样本"（答案里数字较多 / 依据原文很短 / 块本身是表格附近）

运行（crawler 环境）：
    python scripts/12_sample_review.py
产出：
    data/sft/review_40.csv        给你逐条填「判定/编造类型/备注」
    docs/抽检记录.md              记录骨架（含抽样规则与统计口径）
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT, SFT_DIR  # noqa: E402

SEED = 20260924
N = 40
EDGE_N = 5


def main() -> None:
    rows = [json.loads(l) for l in (SFT_DIR / "qa_clean.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if len(rows) < N:
        raise SystemExit(f"❌ 只有 {len(rows)} 条，不够抽 {N} 条")
    rnd = random.Random(SEED)

    by_period: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_period[r["report_period"] + r["report_type"]].append(r)

    picked: list[dict] = []
    seen = set()
    for period, group in sorted(by_period.items()):
        for r in rnd.sample(group, min(3, len(group))):
            picked.append(r)
            seen.add(r["chunk_id"] + r["问题"])

    edge = sorted(rows, key=lambda r: (len(r.get("依据原文") or ""), -len(r["答案"])))[:EDGE_N * 2]
    for r in edge:
        if len(picked) >= N:
            break
        key = r["chunk_id"] + r["问题"]
        if key not in seen:
            picked.append(r)
            seen.add(key)

    rest = [r for r in rows if r["chunk_id"] + r["问题"] not in seen]
    rnd.shuffle(rest)
    picked.extend(rest[:max(0, N - len(picked))])
    picked = picked[:N]
    rnd.shuffle(picked)

    out_csv = SFT_DIR / "review_40.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["序号", "chunk_id", "公司报告期", "章节", "页码", "问题", "答案", "依据原文",
                    "依据是否逐字命中(自动)", "质检有据", "质检自然",
                    "块原文(供核对)", "判定(留/删)", "编造类型", "备注"])
        for i, r in enumerate(picked, 1):
            hit = "是" if r.get("依据原文") and r["依据原文"].strip() and \
                       r["依据原文"].strip() in r["chunk_text"] else "否"
            w.writerow([i, r["chunk_id"], f"{r['company']} {r['report_period']}{r['report_type']}",
                        r["section"], r["printed_page"], r["问题"], r["答案"],
                        r.get("依据原文", ""), hit, r.get("有据"), r.get("自然"),
                        r["chunk_text"][:600], "", "", ""])

    doc = ROOT / "docs" / "抽检记录.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    if not doc.exists():
        doc.write_text(f"""# 作业 B · 人工抽检记录

> 抽样脚本：`scripts/12_sample_review.py`　抽样种子 `random.Random({SEED})`　样本量 {N}

## 抽样规则（可复现）

1. 从 `data/sft/qa_clean.jsonl`（质检通过的问答对）里抽，**不是**从原始生成里抽。
2. 按报告期分层，**每份报告至少 3 条**，保证 2021–2026 各期都被抽到。
3. 额外强制包含 {EDGE_N} 条**边缘样本**（依据原文最短 / 答案最长的那几条）——这些是最可能出编造的地方。
4. 判定标准：**答案里任何一个数字、任何一个事实，只要不能在那条「依据原文」或所在块原文里找到，就算编造，删。**

## 逐条结果

见 `data/sft/review_40.csv`（脚本已把"块原文"填在最后几列，判的时候不用翻 PDF）。
请在 `判定(留/删)`、`编造类型`、`备注` 三列手工填写。

## 统计（抽检完成后填）

| 指标 | 数值 |
|---|---|
| 抽检条数 | {N} |
| 判定"留" | 待填 |
| 判定"删" | 待填 |
| **编造率（删/抽检）** | 待填 |
| 质检阶段二漏掉的编造 | 待填（说明"有据 2"也会漏，人工这关不能省） |

## 典型编造案例（至少 3 条）

1. 待填：原文写的是 ……，模型答成 ……（chunk_id：）
2. 待填
3. 待填

## 处置动作

- 删除了多少条：待填
- 是否回头改了造数据的提示词：待填（改了什么）
- 是否重新生成：待填
""", encoding="utf-8")

    print(f"✅ 抽样 {len(picked)} 条 → {out_csv}")
    print(f"   分层覆盖：{dict(Counter(r['report_period'] + r['report_type'] for r in picked))}")
    print(f"   记录骨架 → {doc}")


if __name__ == "__main__":
    main()
