# -*- coding: utf-8 -*-
"""⑳ 校验 10 道题并冻存哈希（诚实性机制：证明题目是在看到任何结果之前定的）。

产出：
    eval/questions_a.sha256   题目内容的 SHA256（入库）
屏幕：题型覆盖、跨公司题数量、字段完整性检查

运行（crawler 环境）：
    python scripts/20_build_eval_set.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import EVAL_DIR  # noqa: E402

QF = EVAL_DIR / "questions_a.json"
HF = EVAL_DIR / "questions_a.sha256"


def canonical(obj) -> str:
    """只把题目本身（id/type/question/must_refuse）纳入哈希，note 不参与。"""
    core = [{k: q[k] for k in ("id", "type", "question", "must_refuse")} for q in obj]
    return json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    qs = json.loads(QF.read_text(encoding="utf-8"))
    counts = Counter(q["type"] for q in qs)
    scored = [q for q in qs if not q["type"].startswith("附加")]
    cross = [q for q in scored if "跨公司" in q["type"]]

    problems = []
    for q in qs:
        for f in ("id", "type", "question", "must_refuse"):
            if f not in q:
                problems.append(f"{q.get('id','?')} 缺字段 {f}")
    if len(scored) != 10:
        problems.append(f"计入评测的题数是 {len(scored)}，作业要求 10 道")
    if len(cross) < 2:
        problems.append(f"跨公司全景题只有 {len(cross)} 道，作业要求至少 2 道")
    if len({q["question"] for q in qs}) != len(qs):
        problems.append("有重复题目")

    print(f"题目总数 {len(qs)}（计入评测 {len(scored)} + 附加 {len(qs)-len(scored)}）")
    for t, n in counts.items():
        print(f"  {t}: {n}")
    print(f"跨公司全景题：{len(cross)} 道 → {[q['id'] for q in cross]}")

    if problems:
        print("\n❌ 有问题：")
        for p in problems:
            print("   -", p)
        raise SystemExit(1)

    digest = hashlib.sha256(canonical(qs).encode("utf-8")).hexdigest()
    if HF.exists():
        old = HF.read_text(encoding="utf-8").strip().split()[0]
        if old == digest:
            print(f"\n✅ 哈希未变：{digest}")
        else:
            print(f"\n⚠️  题目改过！旧 {old[:16]}… → 新 {digest[:16]}…（诚实起见，改题要在结论里写明）")
    HF.write_text(f"{digest}  frozen at {datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
    print(f"→ 冻存 {HF}")
    # 题目里点了名的公司，供检索 fanout 参考
    names = re.findall(r"(贵州茅台|五粮液|泸州老窖|山西汾酒|洋河股份|古井贡酒|今世缘|口子窖|迎驾贡酒|老白干酒|金种子酒)",
                       " ".join(q["question"] for q in qs))
    print(f"题目涉及公司：{sorted(set(names))}")


if __name__ == "__main__":
    main()
