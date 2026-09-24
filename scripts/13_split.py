# -*- coding: utf-8 -*-
"""⑬ 划分训练/测试集——**按"块（事实）"划分，不按"问题"划分**。

这是四格实验的命门：若按问题划分，同一个事实会同时出现在训练集和测试集，
"没训练过"就是假的，四格的第三、四格会虚假变好（或至少不可信）。

做法：按 chunk_id 分层（报告期 × 章节）85/15 切；训练与评测都只读 split_manifest.json，
物理上杜绝泄漏。

运行（crawler 环境）：
    python scripts/13_split.py
产出：
    data/sft/train.jsonl  data/sft/test.jsonl
    data/sft/split_manifest.json    写死 train_chunk_ids / test_chunk_ids
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SFT_DIR  # noqa: E402

SEED = 20260924
TEST_RATIO = 0.15


def main() -> None:
    rows = [json.loads(l) for l in (SFT_DIR / "qa_clean.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rnd = random.Random(SEED)

    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chunk[r["chunk_id"]].append(r)

    # 分层：报告期 × 章节
    strata: dict[str, list[str]] = defaultdict(list)
    for cid, group in by_chunk.items():
        key = f"{group[0]['report_period']}{group[0]['report_type']}|{group[0]['section']}"
        strata[key].append(cid)

    test_ids, train_ids = [], []
    for key, cids in sorted(strata.items()):
        rnd.shuffle(cids)
        n_test = max(1, round(len(cids) * TEST_RATIO)) if len(cids) >= 3 else 0
        test_ids.extend(cids[:n_test])
        train_ids.extend(cids[n_test:])

    train = [r for cid in train_ids for r in by_chunk[cid]]
    test = [r for cid in test_ids for r in by_chunk[cid]]

    def dump(path: Path, rs: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(SFT_DIR / "train.jsonl", train)
    dump(SFT_DIR / "test.jsonl", test)
    (SFT_DIR / "split_manifest.json").write_text(json.dumps({
        "seed": SEED, "test_ratio": TEST_RATIO, "split_unit": "chunk_id（按事实划分，防泄漏）",
        "train_chunk_ids": sorted(train_ids), "test_chunk_ids": sorted(test_ids),
        "n_train_pairs": len(train), "n_test_pairs": len(test),
        "train_periods": dict(Counter(r["report_period"] + r["report_type"] for r in train)),
        "test_periods": dict(Counter(r["report_period"] + r["report_type"] for r in test)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 训练 {len(train)} 条（{len(train_ids)} 块）｜测试 {len(test)} 条（{len(test_ids)} 块）")
    print(f"   测试集覆盖报告期：{sorted({r['report_period']+r['report_type'] for r in test})}")


if __name__ == "__main__":
    main()
