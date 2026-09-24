# -*- coding: utf-8 -*-
"""⑥ 算向量（Qwen3-Embedding-0.6B，本机 MPS/CPU）。

不走 sentence-transformers：Qwen3-Embedding 是 Qwen3ForCausalLM + **last-token pooling**，
所以用 AutoModel 自己取每条序列最后一个非 pad 位置的 hidden state，再 L2 归一化。
（padding_side="left" 是必须的，否则 last token 取到的是 pad。）

实测：M2 上 MPS 比 CPU 快约 3 倍；8GB 内存下 fp16 权重 1.14GB 完全够。
**必须的工程保护**：分批落盘 + 断点续跑（按 chunk_id 去重）、OOM 自动减 batch。

运行（nlp 环境）：
    python scripts/06_embed.py --set A [--device mps] [--batch 8] [--limit 50]
产出：
    data/index/emb_a.npy      (N, 1024) float32，已归一化
    data/index/emb_a.ids.json 行号 → chunk_id
    data/index/embed_log.json 配置/耗时/tok每秒
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_DIR, EMB_MAX_LEN, EMB_MODEL, HF_ENDPOINT, INDEX_DIR  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Qwen3-Embedding-0.6B"


def load_model(device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = str(MODEL_DIR) if MODEL_DIR.exists() else EMB_MODEL
    print(f"  加载模型：{path}（device={device}）")
    tok = AutoTokenizer.from_pretrained(path, padding_side="left", trust_remote_code=True)
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device).eval()
    return tok, model, torch


def encode(texts: list[str], tok, model, torch, device: str, batch: int, max_len: int) -> np.ndarray:
    """last-token pooling + L2 归一化（tokenizer 必须 padding_side='left'）。"""
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state                       # (B, L, H)
            last = hidden[:, -1]                                          # 左 padding → 末位即真实末 token
            vec = torch.nn.functional.normalize(last.float(), dim=-1)
            out.append(vec.cpu().numpy())
    return np.vstack(out).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B"], default="A")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=EMB_MAX_LEN)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（用于测速）")
    args = ap.parse_args()
    tag = args.set.lower()

    chunk_file = CHUNK_DIR / f"chunks_{tag}.jsonl"
    docs = [json.loads(l) for l in chunk_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        docs = docs[:args.limit]
    print(f"  待嵌入 {len(docs)} 块（max_len={args.max_len}, batch={args.batch}, device={args.device}）")

    emb_path, ids_path = INDEX_DIR / f"emb_{tag}.npy", INDEX_DIR / f"emb_{tag}.ids.json"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    done_ids, done_vecs = [], None
    if emb_path.exists() and ids_path.exists():          # 断点续跑
        done_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        done_vecs = np.load(emb_path)
        print(f"  ⏭  已有 {len(done_ids)} 条，续跑剩余")

    todo = [d for d in docs if d["chunk_id"] not in set(done_ids)]
    if not todo:
        print("✅ 全部已完成")
        return

    device = args.device
    tok, model, torch = load_model(device)
    torch.set_num_threads(max(4, os.cpu_count() or 4))
    texts = [d["index_text"][:3000] for d in todo]       # 超长截断，反正 max_len 会再截

    t0 = time.time()
    vecs = []
    batch = args.batch
    i = 0
    while i < len(texts):
        try:
            v = encode(texts[i:i + batch], tok, model, torch, device, batch, args.max_len)
            vecs.append(v)
            i += batch
            # 每 200 块落一次盘（防"跑了 20 分钟崩了从头再来"）
            if sum(len(x) for x in vecs) % 200 < batch:
                part_new = np.vstack(vecs)
                part_ids = [d["chunk_id"] for d in todo[:len(part_new)]]
                np.save(emb_path, np.vstack([done_vecs, part_new]) if done_vecs is not None and len(done_vecs) else part_new)
                ids_path.write_text(json.dumps((done_ids or []) + part_ids, ensure_ascii=False),
                                    encoding="utf-8")
            if i % (batch * 20) == 0 or i == len(texts) or (i <= batch and args.limit):
                el = time.time() - t0
                done = sum(len(x) for x in vecs)
                speed = done / max(el, 1e-6)
                eta = (len(texts) - done) / max(speed, 1e-6) / 60
                print(f"    {done}/{len(texts)}  {speed:.1f} 块/秒  已用 {el/60:.1f} 分  预计还需 {eta:.1f} 分")
        except RuntimeError as e:                        # 多为 OOM
            if batch > 1:
                batch = max(1, batch // 2)
                print(f"    ⚠️  运行出错（{str(e)[:60]}），batch 降到 {batch} 重试")
                continue                                  # vecs 里已有 i 条结果，直接续跑
            if device == "mps":
                print(f"    ⚠️  MPS 出错（{str(e)[:60]}），退回 CPU fp32 重跑")
                device, batch = "cpu", 4
                tok, model, torch = load_model(device)
                continue
            raise
        if args.limit and i >= args.limit:
            break

    new = np.vstack(vecs) if vecs else np.zeros((0, 1024), dtype="float32")
    ids = [d["chunk_id"] for d in todo[:len(new)]]
    if done_vecs is not None and len(done_vecs):
        new = np.vstack([done_vecs, new])
        ids = done_ids + ids
    np.save(emb_path, new)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    el = time.time() - t0
    log = {"set": tag, "model": EMB_MODEL, "device": device, "batch": batch,
           "max_len": args.max_len, "n_vectors": len(ids), "new_vectors": len(todo[:len(vecs) and len(new)]),
           "seconds": round(el, 1), "chunks_per_sec": round((len(ids) - len(done_ids)) / max(el, 1e-6), 2)}
    (INDEX_DIR / "embed_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(f"\n✅ 向量完成：{len(ids)} 条 × {new.shape[1]} 维 → {emb_path}")


if __name__ == "__main__":
    main()
