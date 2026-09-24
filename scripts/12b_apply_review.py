# -*- coding: utf-8 -*-
"""⑫b 把人工抽检的判定应用到数据上（保留审计痕迹）。

作业 B 第②步要求「人工抽检，删掉编造的」。这里把 `review_40.csv` 里判为「删」的问答对
从 `qa_clean.jsonl` 中剔除，并把判定理由写回抽检表。

**如实说明**（写进 chagelog 与抽检记录）：这 14 条经复核**没有一条是编造**——
它们的「依据原文」逐字命中率 100%、答案里的数字全部能在原文找到；
差别只在**改写幅度**（答案并非逐字引用原文）。所以理由列写的是「改写（非逐字引用）」而不是「编造」。

运行（crawler 环境）：
    python scripts/12b_apply_review.py            # 预演（只打印）
    python scripts/12b_apply_review.py --apply    # 真删
产出：
    data/sft/qa_clean.jsonl                 剔除后的质检通过集合
    data/sft/qa_clean_prereview.jsonl       抽检前的备份（审计用）
    data/sft/review_40.csv                  回填 编造类型 / 备注
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SFT_DIR  # noqa: E402

CLEAN = SFT_DIR / "qa_clean.jsonl"
BACKUP = SFT_DIR / "qa_clean_prereview.jsonl"
REVIEW = SFT_DIR / "review_40.csv"
REASON = "改写（非逐字引用）"
NOTE = "复核：依据原文逐字命中、答案数字全部可在原文找到 → 非编造；仅因未逐字引用而弃用"


def main() -> None:
    apply = "--apply" in sys.argv
    rows = list(csv.DictReader(REVIEW.open(encoding="utf-8")))
    drop_keys = set()
    for r in rows:
        if "删" in (r.get("判定(留/删)") or ""):
            drop_keys.add(r["chunk_id"] + "|" + r["问题"])
            r["编造类型"] = REASON
            r["备注"] = NOTE

    clean = [json.loads(l) for l in CLEAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [c for c in clean if (c["chunk_id"] + "|" + c["问题"]) not in drop_keys]
    n_drop_matched = len(clean) - len(kept)
    print(f"质检通过集：{len(clean)} 条 → 剔除 {n_drop_matched} 条 → 保留 {len(kept)} 条")
    if n_drop_matched != len(drop_keys):
        print(f"  ⚠️ 抽检表里判删 {len(drop_keys)} 条，只在数据里匹配到 {n_drop_matched} 条（可能有重复问题文本）")

    if not apply:
        print("\n（预演模式，未改动任何文件；加 --apply 才真删）")
        return

    if not BACKUP.exists():
        shutil.copy(CLEAN, BACKUP)
        print(f"→ 抽检前备份：{BACKUP}")
    CLEAN.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in kept) + "\n",
                     encoding="utf-8")
    with REVIEW.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"→ {CLEAN} 已更新；判定理由已写回 {REVIEW.name}")
    print("\n下一步（顺序不能反）：13_split.py → 14_make_paraphrase.py → 17_strip_for_git.py "
          "→ 23_local_sft_mps.py 三个阶段重跑")


if __name__ == "__main__":
    main()
