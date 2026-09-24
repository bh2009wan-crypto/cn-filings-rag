# -*- coding: utf-8 -*-
"""⑰ 生成"可以入库"的问答数据版本（去掉块原文，只留问答与短引用）。

为什么不直接入库原始 jsonl：里面带 `chunk_text`（整页财报原文，最长 3000 字/条），
属于课件口径下的"派生文本"。入库版只保留 问题/答案/依据原文（短引用）——
Colab 训练与评测只需要这些，块原文留在本机供质检与抽检用。

运行（crawler 环境）：
    python scripts/17_strip_for_git.py
产出：
    data/sft/git/{qa_clean,train,test,train_para,test_para}.jsonl
    data/sft/git/README.md  （说明这些文件是怎么来的、少了什么）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SFT_DIR  # noqa: E402

DROP = ("chunk_text",)          # 大字段：块原文
KEEP_META = ("chunk_id", "company", "report_period", "report_type", "section", "printed_page",
             "page_range", "问题", "答案", "依据原文", "有据", "自然", "原问题", "_依据等级",
             "_依据覆盖率")


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in DROP and k in KEEP_META}


def main() -> None:
    out = SFT_DIR / "git"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in ("qa_clean", "train", "test", "train_para", "test_para"):
        src = SFT_DIR / f"{name}.jsonl"
        if not src.exists():
            print(f"  ⚠️  缺 {src.name}，跳过")
            continue
        rows = [strip(json.loads(l)) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        (out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        size = (out / f"{name}.jsonl").stat().st_size / 1024
        print(f"  {name}.jsonl：{len(rows)} 条，{size:.0f} KB")
        n += len(rows)

    (out / "README.md").write_text("""# 可入库的问答数据（`data/sft/git/`）

这些文件由 `scripts/17_strip_for_git.py` 从本机的完整版生成，**去掉了 `chunk_text`（整页财报原文）**，
只保留问答与短引用——按"财报 PDF 与派生文本不入库"的约定，块原文留在本机。

| 文件 | 是什么 | 谁用它 |
|---|---|---|
| `qa_clean.jsonl` | 质检通过的问答对（有据 2 且 自然 1） | 交付物 / 审计 |
| `train.jsonl` | 训练集（按**块**划分） | Colab 训练 |
| `test.jsonl` | 测试集（留出的块） | Colab 四格 |
| `train_para.jsonl` | 训练集的换问法 | 四格格2 |
| `test_para.jsonl` | 测试集的换问法 | 四格格4 |

字段：`chunk_id / company / report_period / report_type / section / printed_page / 问题 / 答案 / 依据原文`
（另含质检分 `有据`/`自然`，换问法文件多一个 `原问题`）。

> 想在本机复核"答案是否真有据"，用完整版 `data/sft/*.jsonl`（含 `chunk_text`）——那才是质检与抽检的输入。
""", encoding="utf-8")
    print(f"\n✅ 共 {n} 条 → {out}")


if __name__ == "__main__":
    main()
