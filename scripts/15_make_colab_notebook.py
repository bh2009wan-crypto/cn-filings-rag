# -*- coding: utf-8 -*-
"""⑮ 生成 Colab 笔记本 notebooks/maotai_dongmi_lora.ipynb（作业 B 第③④步）。

笔记本一次跑完：环境 → 探针 → **训练前四格 + RAG 对照** → LoRA SFT → 训练后四格 → 导出结果。
四格 + 1 对照（定义写进 notebook 的 markdown，避免事后解释）：

                原问法      换问法
  训练过的块      格1         格2
  没训练过的块    格3         格4
  RAG 对照                            （同一基座 + A 的知识库检索到的原文）

运行（crawler 环境，只是生成文件）：
    python scripts/15_make_colab_notebook.py
产出：notebooks/maotai_dongmi_lora.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT  # noqa: E402

NB = ROOT / "notebooks" / "maotai_dongmi_lora.ipynb"
REPO = "https://raw.githubusercontent.com/bh2009wan-crypto/homework3/main"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""# 作业 B · 用茅台财报训练一个「董秘」（LoRA SFT）

**做法对齐课件**：造数据（提问/作答/质检三角色同一个便宜模型）→ Colab 免费 T4 上 LoRA 微调 → 四格对照。

## 🚀 怎么跑（三步）

1. 先把 5 个文件上传到这个 notebook 的工作目录（左侧「文件」图标 → 上传；本机路径见下）：
   - `~/work/homework3/data/sft/git/` 下的 `train.jsonl`、`test.jsonl`、`train_para.jsonl`、`test_para.jsonl`
   - `~/work/homework3/data/sft/rag_context.json`（格5 用；不传就跳过格5）
2. 菜单 **代码执行程序 → 更改运行时类型 → 选 T4 GPU**（免费）
3. **全部运行**（Runtime → Run all）。约 **40 分钟**（训练 ~15 分钟 + 前后各一次评估 ~25 分钟）
4. 跑完会自动下载 `results_b.json` / `results_b.csv` / `results_b_summary.md` → 把 `results_b.json` 放到本机 `~/work/homework3/outputs/eval/`，然后执行
   `python scripts/22_run_eval_b.py` 生成 `outputs/B_对照表.md`

**跑挂了怎么办**：
- 第 8 格训练报错（模型不兼容）→ 把第 4 格的 `PRIMARY` 改成 `"Qwen/Qwen3-1.7B"`，重新 Run all
- 报 CUDA out of memory → 第 8 格 `per_device_train_batch_size` 改 1、`gradient_accumulation_steps` 改 16
- 中途断线 → 从头 Run all 即可（训练只要十几分钟）；长跑前建议把适配器目录挂到 Drive

## 四格 + 1 对照（这个定义在跑之前就写死，不许事后改）

|  | 原问法 | 换问法 |
|---|---|---|
| **训练过的块** | 格1 | 格2 |
| **没训练过的块** | 格3 | 格4 |
| **RAG 对照**（同一基座 + 作业 A 的检索） | 格5 | — |

- 划分单位是**块（事实）**不是问题：同一个事实绝不会同时出现在训练集与测试集（`data/sft/split_manifest.json`）。
- 判分：**字符 bigram F1**（主指标）＋ 数字命中（辅助）＋ 误拒答计数。
  ⚠️ 为什么不用"数字命中"当主指标：金答案里 **68% 根本没有数字**（多是"已建成、运行正常"这类叙述），只看数字会全是 0 分。
- 课件的参照结果（它用的是准确率）：训练过原题 2%→100%、训练过换问法 2%→78%、没训练过原题 0%→0%。"""),

    code("""# ── 1. 环境检查 ─────────────────────────────────────────────
!nvidia-smi | head -12
import torch, sys, platform
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")"""),

    code("""# ── 2. 装包（T4 不支持 bf16，训练用 fp16）────────────────────
# 注意：Colab 在国外，**直连 HuggingFace 即可**，不要设 HF_ENDPOINT 镜像（设了反而更慢）；
# 万一拉模型失败，再取消下面这行的注释重跑本格：
# import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
!pip -q install -U "transformers>=5.17" peft trl datasets accelerate
import transformers, peft, trl
print("transformers", transformers.__version__, "| peft", peft.__version__, "| trl", trl.__version__)"""),

    code("""# ── 3. 取数据 ──────────────────────────────────────────────
# ⚠️ 仓库是**私有**的，raw 链接取不到 → 请在 Colab 左侧「文件」面板把这 5 个文件上传到 /content/：
#     data/sft/git/train.jsonl      data/sft/git/test.jsonl
#     data/sft/git/train_para.jsonl data/sft/git/test_para.jsonl
#     data/sft/rag_context.json        ← 本机 scripts/16_dump_rag_for_b.py 生成（格5 用；没有就跳过格5）
# 也可以用 token 拉（可选）：
#     import subprocess; subprocess.run(["git","clone","https://<token>@github.com/bh2009wan-crypto/homework3.git"])
import os, json
FILES = ["train.jsonl", "test.jsonl", "test_para.jsonl", "train_para.jsonl"]
def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []
missing = [f for f in FILES if not os.path.exists(f)]
if missing:
    print("❗ 还没上传这些文件：", missing)
    print("   本机路径：~/work/homework3/data/sft/git/（rag_context.json 在 ~/work/homework3/data/sft/）")
train, test, test_para, train_para = (load(f) for f in FILES)
print({k: len(v) for k, v in zip(["train","test","test_para","train_para"],
                                 [train, test, test_para, train_para])})
if train:
    print("样例：", train[0]["问题"], "→", train[0]["答案"][:60])"""),

    code("""# ── 4. 载模型 ──────────────────────────────────────────────
# 主选 Qwen3.5-2B（课件同款）。⚠️ 它是「视觉+语言」模型且带混合线性注意力，
# LoRA/trl 的兼容性有真实风险 —— 若第 8 格训练报错，把 PRIMARY 改成下面那个再 Run all。
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PRIMARY  = "Qwen/Qwen3.5-2B"      # 课件同款；训练报错就换成 ↓
FALLBACK = "Qwen/Qwen3-1.7B"      # 标准因果 LM，peft/trl 支持最稳

tok = model = None
for name in [PRIMARY, FALLBACK]:
    try:
        print("尝试加载", name, "…")
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                     device_map="auto", trust_remote_code=True)
        print("✅ 成功：", name, "| 参数量 %.2fB" % (sum(p.numel() for p in model.parameters())/1e9))
        break
    except Exception as e:
        print("❌ 失败：", name, type(e).__name__, str(e)[:200])
assert model is not None, "两个候选都没加载成功 —— 换 Qwen/Qwen3-0.6B 再试" """),

    code("""# ── 5. 探针：照实际模块名定 target_modules（不要抄教程）────────
import torch.nn as nn
names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})
print("Linear 层名前缀：", names)
PREF = [x for x in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack") if x in names]
print("将用于 LoRA 的 target_modules：", PREF)
print("（若为空说明模块名特殊，把上面 names 里带 proj/linear 的填进去）")"""),

    code("""# ── 6. 判分与生成工具 ──────────────────────────────────────
import re
from collections import Counter
NEG = re.compile(r"检索到的原文中没有相关内容|材料未涉及|未披露|没有相关")

def numbers(s):
    return {x.replace(",", "").rstrip(".") for x in re.findall(r"\d[\d,]*\.?\d*", s or "")
            if len(x.replace(",", "").replace(".", "")) >= 3 and not re.fullmatch(r"(19|20)\d{2}", x.replace(",", ""))}

def char_grams(s, n=2):
    s = re.sub(r"\s+", "", s or "")
    return [s[i:i+n] for i in range(max(0, len(s) - n + 1))]

def char_f1(pred, gold):
    # 主指标：字符 bigram F1（中文用字级，且不依赖"答案里有没有数字"）
    p, g = char_grams(pred), char_grams(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    prec, rec = common / len(p), common / len(g)
    return round(2 * prec * rec / max(1e-9, prec + rec), 3)

def score(pred, gold):
    g_nums = numbers(gold)
    hit = len(g_nums & numbers(pred)) / len(g_nums) if g_nums else None
    return {
        "char_f1": char_f1(pred, gold),                       # 主指标
        "num_hit": hit,                                        # 辅助：金答案里没数字时为 None
        "len_ratio": round(len(pred or "") / max(1, len(gold)), 2),
        "refused": bool(NEG.search(pred or "")) and not NEG.search(gold or ""),
    }

@torch.inference_mode()
def generate(prompts, max_new=200, bs=8):
    # 批量贪心解码。必须左 padding：右 padding 时短句生成会读到 pad，结果全错。
    out = []
    model.eval()
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        batch = prompts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None,
                             top_p=None, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(batch)):
            out.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip())
    return out

Q_TMPL = "投资者提问：{q}\\n董秘回答："
RAG_TMPL = ("你是贵州茅台董事会秘书。只依据下面材料回答投资者提问，数字要带单位，"
            "材料没有就说没有。\\n材料：{ctx}\\n投资者提问：{q}\\n董秘回答：")

MAX_EVAL = 120      # 每格最多评这么多条（全部跑完约 800 次生成，T4 上 ~30 分钟；想看全量就调大）

def eval_four_grid(tag, rag_ctx=None):
    packs = [("格1_训练过_原问法", train[:MAX_EVAL]), ("格2_训练过_换问法", train_para[:MAX_EVAL]),
             ("格3_没训练过_原问法", test), ("格4_没训练过_换问法", test_para)]
    rows = []
    for name, data in packs:
        if not data:
            continue
        preds = generate([Q_TMPL.format(q=d["问题"]) for d in data])
        for d, pr in zip(data, preds):
            rows.append({"模型": tag, "格": name, "chunk_id": d["chunk_id"], "问题": d["问题"],
                         "预测": pr, "金答案": d["答案"], **score(pr, d["答案"])})
    if rag_ctx:
        inside = [d for d in test[:MAX_EVAL] if d["问题"] in rag_ctx]
        if inside:
            preds = generate([RAG_TMPL.format(ctx=rag_ctx[d["问题"]][:3000], q=d["问题"]) for d in inside])
            for d, pr in zip(inside, preds):
                rows.append({"模型": tag, "格": "格5_RAG对照_测试集", "chunk_id": d["chunk_id"],
                             "问题": d["问题"], "预测": pr, "金答案": d["答案"], **score(pr, d["答案"])})
    return rows"""),

    code("""# ── 7. 训练前基线（四格 + RAG 对照，必须在训练之前跑）─────────
# RAG 上下文：把 A 的知识库检索结果喂进来（没有就跳过，只跑四格）
import os, json
rag_ctx = {}
if os.path.exists("rag_context.json"):
    rag_ctx = json.load(open("rag_context.json", encoding="utf-8"))
    print("载入 RAG 上下文", len(rag_ctx), "条（来自作业 A 的检索）")
else:
    print("没有 rag_context.json —— 格5（RAG 对照）会跳过；本机用 scripts/16_dump_rag_for_b.py 生成")

before = eval_four_grid("训练前", rag_ctx=rag_ctx)
print("训练前完成，", len(before), "条")
import collections
for k, v in collections.Counter((r["格"]) for r in before).items():
    sub = [r for r in before if r["格"] == k]
    print(f"  {k}: {len(sub)} 条，平均数字命中 {sum(x['num_hit'] for x in sub)/len(sub):.2f}")"""),

    code("""# ── 8. 训练（LoRA SFT）─────────────────────────────────────
# 用最稳的写法：数据集只有 text 字段（不用 prompt/completion，避免 trl 版本差异）
# 若报 CUDA out of memory：把 per_device_train_batch_size 改成 1、gradient_accumulation_steps 改成 16
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print("用 target_modules =", PREF)
LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM", target_modules=PREF)
model = get_peft_model(model, LORA)
model.print_trainable_parameters()

ds = Dataset.from_list([{"text": Q_TMPL.format(q=d["问题"]) + d["答案"]} for d in train])
cfg = SFTConfig(output_dir="/content/dongmi_lora", num_train_epochs=3,
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                learning_rate=2e-4, logging_steps=5, save_steps=20, save_total_limit=2,
                fp16=True, gradient_checkpointing=True, max_length=1024,
                dataset_text_field="text", report_to="none", seed=20260924)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
trainer.train()
model.save_pretrained("/content/dongmi_lora/adapter")
tok.save_pretrained("/content/dongmi_lora/adapter")
print("✅ 适配器已存 /content/dongmi_lora/adapter")
# 断线保险：把适配器拷到 Google Drive（挂载后取消注释）
# from google.colab import drive; drive.mount("/content/drive")
# !mkdir -p /content/drive/MyDrive/dongmi && cp -r /content/dongmi_lora/adapter /content/drive/MyDrive/dongmi/"""),

    code("""# ── 9. 训练后四格（与训练前完全同一批题、同一解码参数）──────────
after = eval_four_grid("训练后", rag_ctx=rag_ctx)
print("训练后完成，", len(after), "条")"""),

    code("""# ── 10. 汇总 + 导出 ────────────────────────────────────────
import collections, json, csv
rows = before + after
assert rows, "没有结果 —— 第 7 格的基线评估没跑成功，回去看报错"
with open("results_b.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

g = collections.defaultdict(lambda: {"n": 0, "f1": 0.0, "num": [], "ref": 0, "lr": 0.0})
for r in rows:
    k = (r["模型"], r["格"]); v = g[k]
    v["n"] += 1; v["f1"] += r["char_f1"]; v["ref"] += int(r["refused"]); v["lr"] += r["len_ratio"]
    if r["num_hit"] is not None:
        v["num"].append(r["num_hit"])

lines = ["# 作业 B · 四格对照结果", "",
         "| 模型 | 格 | 题数 | **字符F1(主)** | 数字命中(有数字的题) | 长度比 | 误拒答 |",
         "|---|---|---|---|---|---|---|"]
summary = {}
for (m, cell), v in sorted(g.items()):
    f1 = v["f1"] / max(1, v["n"])
    num = f"{sum(v['num'])/len(v['num']):.2f}（{len(v['num'])} 题）" if v["num"] else "—（该格金答案都没数字）"
    lines.append(f"| {m} | {cell} | {v['n']} | {f1:.3f} | {num} | {v['lr']/max(1,v['n']):.2f} | {v['ref']} |")
    summary[f"{m}|{cell}"] = {"n": v["n"], "char_f1": round(f1, 3),
                              "num_hit": round(sum(v["num"])/len(v["num"]), 3) if v["num"] else None,
                              "n_with_numbers": len(v["num"]), "refused": v["ref"], "len_ratio": round(v["lr"]/max(1,v["n"]), 2)}
lines += ["", "**怎么读**：主指标是字符 bigram F1（金答案里 68% 没有数字，所以不能只看数字命中）。",
          "格1 应明显高于格3（教过的会、没教过的不会）；格2 若明显低于格1，说明背的是问法不是事实；",
          "格5 若不低于格3，说明这类问题该建库而不是微调。"]
open("results_b_summary.md", "w", encoding="utf-8").write("\\n".join(lines))
json.dump({"summary": summary, "n_rows": len(rows), "metric": "char_bigram_f1(+num_hit/refused)"},
          open("results_b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\\n".join(lines))

from google.colab import files
for f in ["results_b.csv", "results_b_summary.md", "results_b.json"]:
    try: files.download(f)
    except Exception as e: print("下载失败", f, e)"""),
]


def main() -> None:
    NB.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 生成 {NB}（{len(CELLS)} 个 cell）")


if __name__ == "__main__":
    main()
