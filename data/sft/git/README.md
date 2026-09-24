# 可入库的问答数据（`data/sft/git/`）

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
