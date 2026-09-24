# -*- coding: utf-8 -*-
"""⑫c 给每条答案补上「依据：《报告名》第 N 页」——对齐课件 B② 与课件的样例格式。

课件原话（p163 作业 B 第②步）：「让大模型从原文生成一批问答：投资者提问 → 董秘作答，
**答案注明出自哪份报告哪一页**」；课件自己的样例答案长这样：
    有责任。根据条例，街道应指导居委会…「…」依据：《上海市医疗保障条例》（第二十三条）
课件在讲到 SFT 教会了什么时也说：「学会**怎么答**：先结论、再引文、**注依据**」。

我们第一版把来源只记在记录元信息里（company/report_period/report_type/printed_page/chunk_id），
答案正文里没有依据——**训出来的模型不会报依据**，这既不符合课件要求，也让四格少了一项可比能力。
这里用**确定性脚本**补上（不需要重跑模型），格式：
    …（答案正文）
    依据：《贵州茅台 2023 年年度报告》第 18 页

运行（crawler 环境）：
    python scripts/12c_add_citation.py            # 预演
    python scripts/12c_add_citation.py --apply    # 写入
产出：就地更新 data/sft/qa_clean.jsonl（幂等：已有「依据：」的不再加）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SFT_DIR  # noqa: E402

CLEAN = SFT_DIR / "qa_clean.jsonl"


def citation_of(rec: dict) -> str:
    period = rec["report_period"]
    rtype = rec["report_type"]
    year = period[:4]
    label = f"《贵州茅台 {year} 年{'年度' if rtype == '年报' else '半年度'}报告》"
    page = rec.get("printed_page")
    return f"依据：{label}第 {page} 页" if page else f"依据：{label}"


def main() -> None:
    apply = "--apply" in sys.argv
    rows = [json.loads(l) for l in CLEAN.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_add = n_skip = 0
    for r in rows:
        if "依据：" in (r["答案"] or ""):
            n_skip += 1
            continue
        r["答案"] = r["答案"].rstrip() + "\n" + citation_of(r)
        n_add += 1
    print(f"共 {len(rows)} 条：新增依据 {n_add} 条，已有依据跳过 {n_skip} 条")
    if rows:
        print("\n样例：")
        for r in rows[:2]:
            print("  问：", r["问题"][:50])
            print("  答：", r["答案"].replace("\n", " ／ ")[:150])
    if not apply:
        print("\n（预演模式，未写入；加 --apply 才生效）")
        return
    CLEAN.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                     encoding="utf-8")
    print(f"→ 已更新 {CLEAN}")


if __name__ == "__main__":
    main()
