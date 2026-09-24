# -*- coding: utf-8 -*-
"""㉓ 本机版「董秘」LoRA：在 Apple M2 上用 MPS 跑同一套四格 + RAG 对照。

为什么有这个脚本：Colab 那条路被网络卡死（代理 20 分钟内三次掉线，Colab 的长连接撑不住）。
本机版**完全不依赖网络**，代价是基座换成 **Qwen3-0.6B**（课件用的是 Qwen3.5-2B，8GB 内存装不下 2B 的训练）。
四格的定义、判分口径与 `notebooks/maotai_dongmi_lora.ipynb` **完全一致**，两个结果可以直接对比。

四格 + 1 对照：
    训练过的块 × 原问法/换问法 = 格1/格2
    没训练过的块 × 原问法/换问法 = 格3/格4
    格5 = 同一基座 + 作业 A 的检索原文（测试集上）

判分：字符 bigram F1（主）＋ 数字命中（辅助）＋ 误拒答计数。
（不是"数字命中"当主指标：金答案里 68% 根本没有数字。）

运行（nlp 环境）：
    python scripts/23_local_sft_mps.py --cap 60 --epochs 3          # 正式
    python scripts/23_local_sft_mps.py --smoke                      # 冒烟测试（3 步训练 + 5 条评估）
产出：
    outputs/eval/results_b.csv           逐题记录
    outputs/eval/results_b.json          汇总（给 22_run_eval_b.py 用）
    outputs/eval/local_adapter/          训练好的 LoRA 适配器
    logs/23_local_sft.log                训练与评估日志
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUT_DIR, ROOT, SFT_DIR  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen3-0.6B"
GIT_DIR = SFT_DIR / "git"
OUT = OUT_DIR / "eval"

# ---------- 判分（与笔记本逐字一致） ----------
NEG = re.compile(r"检索到的原文中没有相关内容|材料未涉及|未披露|没有相关")


def numbers(s: str) -> set[str]:
    return {x.replace(",", "").rstrip(".") for x in re.findall(r"\d[\d,]*\.?\d*", s or "")
            if len(x.replace(",", "").replace(".", "")) >= 3
            and not re.fullmatch(r"(19|20)\d{2}", x.replace(",", ""))}


def char_grams(s: str, n: int = 2) -> list[str]:
    s = re.sub(r"\s+", "", s or "")
    return [s[i:i + n] for i in range(max(0, len(s) - n + 1))]


def char_f1(pred: str, gold: str) -> float:
    p, g = char_grams(pred), char_grams(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    prec, rec = common / len(p), common / len(g)
    return round(2 * prec * rec / max(1e-9, prec + rec), 3)


def score(pred: str, gold: str) -> dict:
    gn = numbers(gold)
    return {
        "char_f1": char_f1(pred, gold),
        "num_hit": (len(gn & numbers(pred)) / len(gn)) if gn else None,
        "len_ratio": round(len(pred or "") / max(1, len(gold)), 2),
        "refused": bool(NEG.search(pred or "")) and not NEG.search(gold or ""),
    }


Q_TMPL = "投资者提问：{q}\n董秘回答："
RAG_TMPL = ("你是贵州茅台董事会秘书。只依据下面材料回答投资者提问，数字要带单位，"
            "材料没有就说没有，并在末尾用「依据：《报告名》第 N 页」注明出处。"
            "\n材料：{ctx}\n投资者提问：{q}\n董秘回答：")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if p.exists() else []


class Boss:
    """模型 + 分词 + 生成 + 训练，全在一个类里，方便冒烟测试与正式跑共用。"""

    def __init__(self, model_path: str, device: str = "mps", dtype: str = "bf16"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device if (device != "mps" or torch.backends.mps.is_available()) else "cpu"
        # 实测：M2 上 bf16 比 fp32 快约 11 倍（每 batch 0.9s vs 10.0s）。
        # peft 会把 LoRA 适配器保持 fp32，所以 bf16 主干 + fp32 适配器既快又稳。
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        print(f"  加载 {model_path}（device={self.device}，主干 {dtype} / LoRA 适配器 fp32）…")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dt, trust_remote_code=True).to(self.device)
        self.model.eval()
        print(f"  ✅ 加载完成 {time.time()-t0:.0f}s｜参数量 "
              f"{sum(p.numel() for p in self.model.parameters())/1e9:.2f}B")

    # ---------- 生成 ----------
    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new: int = 160, bs: int = 4) -> list[str]:
        self.model.eval()
        self.tok.padding_side = "left"          # 关键：右 padding 会让短句读到 pad，结果全错
        out = []
        for i in range(0, len(prompts), bs):
            batch = prompts[i:i + bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=768).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            for j in range(len(batch)):
                out.append(self.tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True).strip())
        return out

    # ---------- 训练 ----------
    def train_lora(self, pairs: list[tuple[str, str]], epochs: int = 3, lr: float = 2e-4,
                   bs: int = 1, accum: int = 16, max_len: int = 192, steps_cap: int = 0) -> dict:
        from peft import LoraConfig, get_peft_model
        names = sorted({n.split(".")[-1] for n, m in self.model.named_modules()
                        if isinstance(m, torch.nn.Linear)})
        targets = [x for x in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                               "up_proj", "down_proj") if x in names]
        print(f"  探针：Linear 层名 {names[:12]}{'…' if len(names) > 12 else ''}")
        assert targets, f"没找到可用的 target_modules（实际层名：{names}）"
        print(f"  LoRA target_modules = {targets}")

        self.model = get_peft_model(self.model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=targets))
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  可训练参数 {trainable/1e6:.2f}M / 总 {total/1e9:.2f}B（{trainable/total:.2%}）")

        # 预编码（prompt 部分 label 置 -100，只在答案上算损失）
        samples = []
        for q, a in pairs:
            p_ids = self.tok(Q_TMPL.format(q=q), add_special_tokens=False)["input_ids"]
            f_ids = self.tok(Q_TMPL.format(q=q) + a + self.tok.eos_token,
                             add_special_tokens=False)["input_ids"][:max_len]
            labels = list(f_ids)
            for k in range(min(len(p_ids), len(labels))):
                labels[k] = -100
            samples.append((f_ids, labels))

        opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.model.train()
        step, log = 0, []
        t0 = time.time()
        for ep in range(epochs):
            for i in range(0, len(samples), bs):
                batch = samples[i:i + bs]
                maxlen = max(len(x[0]) for x in batch)
                ids = torch.full((len(batch), maxlen), self.tok.pad_token_id, dtype=torch.long)
                lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
                for r, (f, l) in enumerate(batch):
                    ids[r, :len(f)] = torch.tensor(f)
                    lab[r, :len(l)] = torch.tensor(l)
                loss = self.model(input_ids=ids.to(self.device),
                                  labels=lab.to(self.device)).loss / accum
                loss.backward()
                if (i // bs + 1) % accum == 0 or i + bs >= len(samples):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if hasattr(torch, "mps") and torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    if step % 5 == 0 or step == 1:
                        print(f"    ep{ep+1} step{step} loss={loss.item()*accum:.4f} "
                              f"({time.time()-t0:.0f}s)")
                    log.append({"step": step, "loss": round(loss.item() * accum, 4)})
                    if steps_cap and step >= steps_cap:
                        print(f"    （冒烟测试：到 {steps_cap} 步停）")
                        return {"steps": step, "seconds": round(time.time() - t0, 1),
                                "loss_log": log, "trainable": trainable, "targets": targets}
        self.model.eval()
        return {"steps": step, "seconds": round(time.time() - t0, 1), "loss_log": log,
                "trainable": trainable, "targets": targets}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tok.save_pretrained(str(path))


def eval_grid(boss: Boss, tag: str, train, test, train_para, test_para,
              rag_ctx: dict, cap: int, max_new: int) -> list[dict]:
    packs = [("格1_训练过_原问法", train[:cap]), ("格2_训练过_换问法", train_para[:cap]),
             ("格3_没训练过_原问法", test[:cap]), ("格4_没训练过_换问法", test_para[:cap])]
    rows = []
    for name, data in packs:
        if not data:
            continue
        t0 = time.time()
        preds = boss.generate([Q_TMPL.format(q=d["问题"]) for d in data], max_new=max_new)
        for d, pr in zip(data, preds):
            rows.append({"模型": tag, "格": name, "chunk_id": d["chunk_id"], "问题": d["问题"],
                         "预测": pr, "金答案": d["答案"], **score(pr, d["答案"])})
        print(f"    {name}: {len(data)} 条，{time.time()-t0:.0f}s")
    inside = [d for d in test[:cap] if d["问题"] in rag_ctx]
    if inside:
        t0 = time.time()
        preds = boss.generate([RAG_TMPL.format(ctx=rag_ctx[d["问题"]][:2500], q=d["问题"])
                               for d in inside], max_new=max_new)
        for d, pr in zip(inside, preds):
            rows.append({"模型": tag, "格": "格5_RAG对照_测试集", "chunk_id": d["chunk_id"],
                         "问题": d["问题"], "预测": pr, "金答案": d["答案"],
                         **score(pr, d["答案"])})
        print(f"    格5_RAG对照: {len(inside)} 条，{time.time()-t0:.0f}s")
    return rows


def summarize(rows: list[dict]) -> tuple[str, dict]:
    g = {}
    for r in rows:
        k = (r["模型"], r["格"])
        v = g.setdefault(k, {"n": 0, "f1": 0.0, "num": [], "ref": 0, "lr": 0.0})
        v["n"] += 1; v["f1"] += r["char_f1"]; v["ref"] += int(r["refused"]); v["lr"] += r["len_ratio"]
        if r["num_hit"] is not None:
            v["num"].append(r["num_hit"])
    lines = ["# 作业 B · 四格对照结果（本机 M2 / Qwen3-0.6B）", "",
             "| 模型 | 格 | 题数 | **字符F1(主)** | 数字命中(有数字的题) | 长度比 | 误拒答 |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for (m, cell), v in sorted(g.items()):
        f1 = v["f1"] / max(1, v["n"])
        num = f"{sum(v['num'])/len(v['num']):.2f}（{len(v['num'])} 题）" if v["num"] else "—"
        lines.append(f"| {m} | {cell} | {v['n']} | {f1:.3f} | {num} | {v['lr']/max(1,v['n']):.2f} | {v['ref']} |")
        summary[f"{m}|{cell}"] = {"n": v["n"], "char_f1": round(f1, 3),
                                  "num_hit": round(sum(v["num"]) / len(v["num"]), 3) if v["num"] else None,
                                  "n_with_numbers": len(v["num"]), "refused": v["ref"],
                                  "len_ratio": round(v["lr"] / max(1, v["n"]), 2)}
    lines += ["", "**怎么读**：主指标是字符 bigram F1；格1 应明显高于格3（教过的会、没教过的不会）；",
              "格2 若明显低于格1，说明背的是问法不是事实；格5 若不低于格3，说明这类问题该建库而不是微调。"]
    return "\n".join(lines), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60, help="每格最多评估多少条")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="主干精度；M2 上 bf16 比 fp32 快约 11 倍（实测）")
    ap.add_argument("--model", default=str(MODEL_DIR))
    ap.add_argument("--phase", default="all",
                    choices=["all", "before", "train", "after", "merge"],
                    help="分阶段跑：8GB 内存机器上建议 before / train / after 分三个进程，"
                         "避免评估与训练的内存叠在一起触发 MPS OOM")
    ap.add_argument("--bs", type=int, default=1, help="训练 batch（8GB 机器建议 1）")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192, help="训练序列上限（实测 p90=142 token）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：3 步训练 + 每格 5 条")
    args = ap.parse_args()
    if args.smoke:
        args.cap, args.epochs = 5, 1

    OUT.mkdir(parents=True, exist_ok=True)
    train = load_jsonl(GIT_DIR / "train.jsonl")
    test = load_jsonl(GIT_DIR / "test.jsonl")
    train_para = load_jsonl(GIT_DIR / "train_para.jsonl")
    test_para = load_jsonl(GIT_DIR / "test_para.jsonl")
    rag = json.loads((SFT_DIR / "rag_context.json").read_text(encoding="utf-8")) \
        if (SFT_DIR / "rag_context.json").exists() else {}
    print(f"  数据：train {len(train)}｜test {len(test)}｜train_para {len(train_para)}"
          f"｜test_para {len(test_para)}｜RAG {len(rag)}")
    if not train:
        raise SystemExit(f"❌ 找不到训练数据（{GIT_DIR}）")

    before_f, after_f = OUT / "rows_before.json", OUT / "rows_after.json"
    adapter = OUT / "local_adapter"
    info: dict = {}

    # ---------- 阶段 1：训练前四格 ----------
    if args.phase in ("all", "before"):
        boss = Boss(args.model, args.device, args.dtype)
        print("\n=== 训练前：四格 + RAG 对照 ===")
        before = eval_grid(boss, "训练前", train, test, train_para, test_para, rag,
                           args.cap, args.max_new)
        before_f.write_text(json.dumps(before, ensure_ascii=False), encoding="utf-8")
        print(f"  → 已存 {before_f}")
        del boss
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if args.phase == "before":
            print("\n（--phase before 完成）接下来跑：--phase train")
            return

    # ---------- 阶段 2：训练 ----------
    if args.phase in ("all", "train"):
        boss = Boss(args.model, args.device, args.dtype)
        print("\n=== LoRA 训练 ===")
        info = boss.train_lora([(d["问题"], d["答案"]) for d in train], epochs=args.epochs,
                               bs=args.bs, accum=args.accum, max_len=args.max_len,
                               steps_cap=(3 if args.smoke else 0))
        print(f"  训练完成：{info['steps']} 步 / {info['seconds']}s")
        boss.save(adapter)
        (OUT / "train_info.json").write_text(json.dumps(
            {**{k: v for k, v in info.items() if k not in ("trainable", "targets")},
             "trainable_params": info.get("trainable"), "target_modules": info.get("targets"),
             "loss_first": (info.get("loss_log") or [{}])[0].get("loss"),
             "loss_last": (info.get("loss_log") or [{}])[-1].get("loss"),
             "source": f"{args.device}/{args.dtype}", "epochs": args.epochs,
             "lr": 2e-4, "lora_r": 16, "lora_alpha": 32,
             "batch": args.bs, "grad_accum": args.accum, "max_len": args.max_len},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → 适配器已存 {adapter}；训练信息 {OUT/'train_info.json'}")
        del boss
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if args.phase == "train":
            print("\n（--phase train 完成）接下来跑：--phase after")
            return

    # ---------- 阶段 3：训练后四格（加载适配器） ----------
    if args.phase in ("all", "after"):
        boss = Boss(args.model, args.device, args.dtype)
        if adapter.exists():
            from peft import PeftModel
            boss.model = PeftModel.from_pretrained(boss.model, str(adapter))
            boss.model.eval()
            print(f"  ✅ 已加载适配器 {adapter}")
        else:
            print(f"  ⚠️  没有适配器（{adapter}），训练后四格等同于基座复测")
        print("\n=== 训练后：四格 + RAG 对照 ===")
        after = eval_grid(boss, "训练后", train, test, train_para, test_para, rag,
                          args.cap, args.max_new)
        after_f.write_text(json.dumps(after, ensure_ascii=False), encoding="utf-8")
        print(f"  → 已存 {after_f}")
        del boss
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if args.phase == "after":
            print("\n（--phase after 完成）接下来跑：--phase merge")
            return

    # ---------- 阶段 4：汇总 ----------
    before = json.loads(before_f.read_text(encoding="utf-8")) if before_f.exists() else []
    after = json.loads(after_f.read_text(encoding="utf-8")) if after_f.exists() else []
    rows = before + after
    if not rows:
        raise SystemExit("❌ 没有结果可汇总（先跑 --phase before / after）")
    with (OUT / "results_b.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    table, summary = summarize(rows)
    (OUT / "results_b_summary.md").write_text(table, encoding="utf-8")
    (OUT / "results_b.json").write_text(json.dumps(
        {"summary": summary, "n_rows": len(rows), "metric": "char_bigram_f1(+num_hit/refused)",
         "base_model": Path(args.model).name, "device": args.device, "dtype": args.dtype,
         "train_info": (info or (json.loads((OUT / "train_info.json").read_text(encoding="utf-8"))
                                 if (OUT / "train_info.json").exists() else {})),
         "cap_per_cell": args.cap, "loss_log": (info.get("loss_log") or [])[-20:]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + table)
    print(f"\n→ {OUT/'results_b.csv'}｜{OUT/'results_b_summary.md'}｜{OUT/'results_b.json'}")


if __name__ == "__main__":
    main()
