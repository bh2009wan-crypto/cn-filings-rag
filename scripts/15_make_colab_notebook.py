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

## 四格 + 1 对照（这个定义在跑之前就写死，不许事后改）

|  | 原问法 | 换问法 |
|---|---|---|
| **训练过的块** | 格1 | 格2 |
| **没训练过的块** | 格3 | 格4 |
| **RAG 对照**（同一基座 + A 的知识库） | 格5a | 格5b |

- 划分单位是**块（事实）**不是问题：同一个事实绝不会同时出现在训练集与测试集（见 `data/sft/split_manifest.json`）。
- 判分：先用 `key_facts` 做**确定性**数字命中（可复现），再给一个 LLM 判分（可选）。
- 课件的参照结果：训练过原题 2%→100%、训练过换问法 2%→78%、没训练过原题 0%→0%、没训练过换问法 0%→0%。

⚠️ 跑完请把 `results_b.json` / `results_b_summary.md` 下载回本机，放进 `outputs/eval/`。"""),

    code("""# ── 1. 环境检查 ─────────────────────────────────────────────
!nvidia-smi | head -12
import torch, sys, platform
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")"""),

    code("""# ── 2. 装包（先设镜像；T4 不支持 bf16，训练用 fp16）──────────
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"     # HuggingFace 被墙，走镜像
!pip -q install -U "transformers>=5.17" peft trl datasets accelerate bitsandbytes
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

    code("""# ── 4. 载模型（主选 Qwen3.5-2B＝课件同款；失败自动退 Qwen3-1.7B）──
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
CANDIDATES = ["Qwen/Qwen3.5-2B", "Qwen/Qwen3-1.7B"]
tok = model = None
for name in CANDIDATES:
    try:
        print("尝试加载", name, "…")
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                     device_map="auto", trust_remote_code=True)
        print("✅ 成功：", name)
        break
    except Exception as e:
        print("❌ 失败：", name, type(e).__name__, str(e)[:200])
assert model is not None, "两个候选模型都没加载成功，请手动换一个更小的 Qwen 模型" """),

    code("""# ── 5. 探针：照实际模块名定 target_modules（不要抄教程）────────
import torch.nn as nn
names = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})
print("Linear 层名前缀：", names)
PREF = [x for x in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack") if x in names]
print("将用于 LoRA 的 target_modules：", PREF)
print("（若为空说明模块名特殊，把上面 names 里带 proj/linear 的填进去）")"""),

    code("""# ── 6. 判分与生成工具 ──────────────────────────────────────
import re, json
NEG = re.compile(r"检索到的原文中没有相关内容|材料未涉及|未披露|没有相关")

def numbers(s):
    return {x.replace(",", "").rstrip(".") for x in re.findall(r"\\d[\\d,]*\\.?\\d*", s or "")
            if len(x.replace(",", "").replace(".", "")) >= 3}

def score(pred, gold_answer, gold_evidence):
    # 确定性判分：金答案里的数字命中率；并检查是否"误拒答"
    g, p = numbers(gold_answer), numbers(pred)
    hit = len(g & p) / max(1, len(g))
    refused = bool(NEG.search(pred or "")) and not NEG.search(gold_answer or "")
    return {"num_hit": round(hit, 3), "refused": refused, "n_gold_nums": len(g)}

@torch.inference_mode()
def generate(prompts, max_new=220, bs=8):
    out = []
    model.eval()
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

def eval_four_grid(tag, rag_ctx=None):
    \"\"\"跑四格（+RAG 对照）并返回逐题记录。rag_ctx: {问题: 检索原文} 或 None\"\"\"
    packs = [("格1_训练过_原问法", train), ("格2_训练过_换问法", train_para),
             ("格3_没训练过_原问法", test), ("格4_没训练过_换问法", test_para)]
    rows = []
    for name, data in packs:
        if not data:
            continue
        prompts = [Q_TMPL.format(q=d["问题"]) for d in data]
        preds = generate(prompts)
        for d, pr in zip(data, preds):
            rows.append({"模型": tag, "格": name, "chunk_id": d["chunk_id"],
                         "问题": d["问题"], "预测": pr, "金答案": d["答案"],
                         "依据原文": d.get("依据原文", ""),
                         **score(pr, d["答案"], d.get("依据原文", ""))})
    if rag_ctx:
        prompts, keys = [], []
        for d in test:
            ctx = rag_ctx.get(d["问题"])
            if ctx:
                prompts.append(RAG_TMPL.format(ctx=ctx[:3000], q=d["问题"])); keys.append(d)
        if prompts:
            preds = generate(prompts)
            for d, pr in zip(keys, preds):
                rows.append({"模型": tag, "格": "格5_RAG对照_测试集", "chunk_id": d["chunk_id"],
                             "问题": d["问题"], "预测": pr, "金答案": d["答案"],
                             "依据原文": d.get("依据原文", ""),
                             **score(pr, d["答案"], d.get("依据原文", ""))})
    return rows"""),

    code("""# ── 7. 训练前基线（四格 + RAG 对照，必须在训练之前跑）─────────
# RAG 上下文：把 A 的知识库检索结果喂进来（没有就跳过，只跑四格）
import os, json
rag_ctx = {}
if os.path.exists("rag_context.json"):
    rag_ctx = json.load(open("rag_context.json", encoding="utf-8"))
    print("载入 RAG 上下文", len(rag_ctx), "条（来自作业 A 的检索）")
else:
    print("没有 rag_context.json —— 格5 会跳过（本机跑 21_run_eval_a.py --dump-rag 可生成）")

before = eval_four_grid("训练前", rag_ctx=rag_ctx)
print("训练前完成，", len(before), "条")
import collections
for k, v in collections.Counter((r["格"]) for r in before).items():
    sub = [r for r in before if r["格"] == k]
    print(f"  {k}: {len(sub)} 条，平均数字命中 {sum(x['num_hit'] for x in sub)/len(sub):.2f}")"""),

    code("""# ── 8. 训练（LoRA SFT）─────────────────────────────────────
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

LORA = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  task_type="CAUSAL_LM", target_modules=PREF)
model = get_peft_model(model, LORA)
model.print_trainable_parameters()

ds = Dataset.from_list([{"prompt": Q_TMPL.format(q=d["问题"]), "completion": d["答案"]}
                        for d in train])
cfg = SFTConfig(output_dir="/content/dongmi_lora", num_train_epochs=3,
                per_device_train_batch_size=2, gradient_accumulation_steps=8,
                learning_rate=2e-4, logging_steps=10, save_steps=100, save_total_limit=2,
                fp16=True, gradient_checkpointing=True, max_length=1024,
                completion_only_loss=True, report_to="none", seed=20260924)
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
with open("results_b.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

grid = collections.defaultdict(lambda: {"n": 0, "hit": 0.0, "refused": 0})
for r in rows:
    k = (r["模型"], r["格"])
    grid[k]["n"] += 1; grid[k]["hit"] += r["num_hit"]; grid[k]["refused"] += int(r["refused"])
lines = ["# 作业 B · 四格对照结果", "",
         "| 模型 | 格 | 题数 | 平均数字命中率 | 误拒答数 |", "|---|---|---|---|---|"]
summary = {}
for (m, g), v in sorted(grid.items()):
    h = v["hit"] / max(1, v["n"])
    lines.append(f"| {m} | {g} | {v['n']} | {h:.2f} | {v['refused']} |")
    summary[f"{m}|{g}"] = {"n": v["n"], "num_hit": round(h, 3), "refused": v["refused"]}
open("results_b_summary.md", "w", encoding="utf-8").write("\\n".join(lines))
json.dump({"summary": summary, "n_rows": len(rows), "model": model.config._name_or_path if hasattr(model,'config') else "?"},
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
