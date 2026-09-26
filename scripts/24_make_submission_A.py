# -*- coding: utf-8 -*-
"""㉔ 生成作业 3「方向 A」的提交包（交平台用）。

课件 p163 要求方向 A 交三样：**代码仓库 + 页面截图 + 一页结论**。
本脚本把它们合成一个直接可粘的目录：

    提交_A/
        作业3_A_提交材料.md          ← 粘到平台；= 仓库链接 + 截图说明 + 一页结论**全文**
        A_页面截图_1_首页.png
        A_页面截图_2_回答与证据卡片.png

「一页结论」是从 docs/结论A_一页.md **原文拼入**的（不手抄），所以结论改了重跑本脚本即可。
提交包是生成物，放在 .gitignore 里、不随仓库公开。

跑法（crawler 环境，只是拷文件+拼文本）：
    python scripts/24_make_submission_A.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT  # noqa: E402

OUT = ROOT / "提交_A"
CONCLUSION = ROOT / "docs" / "结论A_一页.md"
SHOTS = [
    ("outputs/A_页面截图.png", "A_页面截图_1_首页.png"),
    ("outputs/A_页面截图_回答.png", "A_页面截图_2_回答与证据卡片.png"),
]

HEADER = """# 作业 3 · 方向 A：白酒 11 家财报问答知识库

> 课件《学会使用大模型 · 第三课》p163「作业 3：两个方向，任选一个」——本份提交走 **方向 A**（下载财报做问答知识库）。
> 语料：白酒 11 家上市公司的 2026 年半年报**全文**，取自**巨潮资讯网**（交易所指定信息披露平台）。

## 一、代码仓库

**https://github.com/bh2009wan-crypto/cn-filings-rag**

仓库**已公开**，无需登录即可访问、可复现。方向 A 的实现是 `scripts/01–09`
（抓取 → PDF 解析 → 切块 → 向量+BM25 双索引 → 混合检索 → 问答页面）；
「课件交付物 ↔ 仓库文件」的逐项对照表在仓库 `README.md`。

## 二、页面截图（随本材料上传两张）

| 文件 | 内容 |
|---|---|
| `A_页面截图_1_首页.png` | 问答页面首页：提问框 + 11 道题的按钮 |
| `A_页面截图_2_回答与证据卡片.png` | 一次真实回答：答案带 `[n]` 角标、证据卡片给出「公司·章节·页码·chunk_id」、页面上直接显示数字核对结果 |

> 第二张**故意选的是答错的那道全景题（Q9）**，不是挑一张好看的：画面里「数字核对 通过」，但结论是错的
> （金种子酒取成了母公司口径）。它正好是下面一页结论第五节那个判断的现场证据——
> **数字核对只能防编造，防不住口径错**。

## 三、一页结论

<!-- 以下为 docs/结论A_一页.md 原文，由 scripts/24_make_submission_A.py 自动拼入 -->

"""

FOOTER = """

---

## 附：怎么把问答页面跑起来

复现步骤（环境、依赖、启动命令）见仓库 `README.md`；页面起来后点题目按钮即可复现这 11 道题。
"""


def main() -> None:
    if not CONCLUSION.exists():
        raise SystemExit(f"❌ 找不到 {CONCLUSION}")
    OUT.mkdir(exist_ok=True)

    body = CONCLUSION.read_text(encoding="utf-8")
    md = OUT / "作业3_A_提交材料.md"
    md.write_text(HEADER + body + FOOTER, encoding="utf-8")

    for src_rel, dst_name in SHOTS:
        src = ROOT / src_rel
        if not src.exists():
            raise SystemExit(f"❌ 找不到截图 {src}")
        shutil.copy2(src, OUT / dst_name)

    print(f"✅ 提交包已生成：{OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"   {p.name}")
    print(f"\n一页结论 {len(body.splitlines())} 行（拼自 {CONCLUSION.name}）→ 直接粘「三、一页结论」那节到平台即可。")


if __name__ == "__main__":
    main()
