# -*- coding: utf-8 -*-
"""检索 + 作答核心库（页面 08_app.py 与评测 21_run_eval_a.py 共用同一条链）。

检索：BM25 top-k ∪ 向量 top-k → RRF 融合 → top-N（同一张表最多 2 段，避免一张大表刷屏）。
     跨公司问题走 fanout：对 11 家各跑一次限定公司的检索，每家取 2 块——
     全局 top-8 不可能覆盖 11 家，全景题会必然翻车（课件：Agentic 挑到 11 家却放不下 4 家）。
作答：deepseek-flash，硬约束「只许引用给定块、每句带 [n] 角标、原文没有就说没有」，
     **并在提示词里注入今天的日期**（实测模型内部日期是 2026-05-07，会一口咬定 2026 半年报"尚未披露"）。
后置校验：角标合法性 + 答案里的数字能否在检索块里找到（把"数字纪律"变成可测指标）。

命令行入口见 scripts/07_ask.py（薄封装）。
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (BM25_TOP, CHUNK_DIR, COMPANY_B, COMPANIES, EMB_MAX_LEN, FANOUT_PER_COMPANY,  # noqa: E402
                    FINAL_TOP, HF_ENDPOINT, INDEX_DIR, LLM_MODEL, RRF_K, TODAY, VEC_TOP)
from utils.llm import chat  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)

SYSTEM = """你是白酒行业上市公司财报的问答助手。规则（必须逐条遵守）：
1. 今天是 2026年9月24日。**不要用你内部的记忆判断这些报告是否已发布、是否存在于你的知识里**——你面对的原文就是事实。
2. 只依据下面【检索到的原文】作答，禁止使用任何外部知识或推测。
3. 每个数字、每个结论后面必须紧跟出处角标，形如 [1]、[2]（可多个，如 [1][3]）。
4. 原文里没有的，直接回答「检索到的原文中没有相关内容」，不要猜、不要编。
5. 涉及对比或排名时，先把各家的原始数字与单位（元/亿元/万元）逐个列出，再给结论。
6. 涉及财务口径时必须写清是「合并报表」还是「母公司报表」、以及报告期。
7. 简洁作答，不超过 300 字；不要输出 markdown 标题。"""

STRICT_TAIL = """
【检索到的原文】
{context}

【问题】{q}

请按规则作答（每个数字后带 [n] 角标）。"""


def _num_tokens(s: str) -> tuple[set[str], set[str]]:
    """抽出答案里的数字，分成两组：

    · 需要核对的：原样引用财报的数字（必须能在检索块里找到）
    · 豁免的：4 位年份（19xx/20xx）、以及答案自己算出来的数（前面有 ≈/约/为 … 计算值）

    实测踩到过：`2026年上半年` 里的 2026、以及 `≈41.44%` 这种**计算出来**的比率，
    都会被朴素核对判成"原文没有"，把好答案误标成可疑。这里显式区分，而不是放宽阈值。
    """
    need, exempt = set(), set()
    for m in re.finditer(r"(≈|约|算出|计算得|得到)?\s*(\d[\d,]*\.?\d*)", s):
        prefix, raw = m.group(1), m.group(2)
        t = raw.replace(",", "").rstrip(".")
        if len(t.replace(".", "")) < 3:
            continue
        if re.fullmatch(r"(19|20)\d{2}", t):       # 年份
            exempt.add(t)
            continue
        (exempt if prefix else need).add(t)
    return need, exempt


class Retriever:
    def __init__(self, tag: str = "a"):
        self.tag = tag
        self.docs = [json.loads(l) for l in
                     (CHUNK_DIR / f"chunks_{tag}.jsonl").read_text(encoding="utf-8").splitlines()
                     if l.strip()]
        self.by_id = {d["chunk_id"]: d for d in self.docs}
        self.bm = pickle.loads((INDEX_DIR / f"bm25_{tag}.pkl").read_bytes())
        self.emb = np.load(INDEX_DIR / f"emb_{tag}.npy")
        self.ids = json.loads((INDEX_DIR / f"emb_{tag}.ids.json").read_text(encoding="utf-8"))
        self.row_of = {cid: i for i, cid in enumerate(self.ids)}
        self._tok = None
        self._model = None

    # ---------- 分词 / 向量 ----------
    def _tokenize(self, s: str):
        if self._tok is None:
            try:
                import jieba
                jieba.initialize()
                self._tok = lambda x: [w for w in jieba.lcut(x) if w.strip()]
            except Exception:
                self._tok = lambda x: [x[i:i + 2] for i in range(max(0, len(x) - 1))]
        return self._tok(s)

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            p = Path(__file__).resolve().parent.parent / "models" / "Qwen3-Embedding-0.6B"
            path = str(p) if p.exists() else "Qwen/Qwen3-Embedding-0.6B"
            tok = AutoTokenizer.from_pretrained(path, padding_side="left", trust_remote_code=True)
            model = AutoModel.from_pretrained(path, torch_dtype=torch.float16,
                                              trust_remote_code=True).eval()
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            model = model.to(dev)
            self._model = (tok, model, torch, dev)
        tok, model, torch, dev = self._model
        with torch.inference_mode():
            enc = tok([text], padding=True, truncation=True, max_length=EMB_MAX_LEN,
                      return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state[:, -1]
            v = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()[0]
        return v

    def _query_text(self, q: str) -> str:
        return f"Instruct: 检索白酒上市公司半年报中与问题相关的原文\nQuery: {q}"

    # ---------- 两路召回 ----------
    def _bm25(self, q: str, top: int) -> list[tuple[str, float]]:
        from collections import Counter
        idf, k1, b, avgdl = self.bm["idf"], self.bm["k1"], self.bm["b"], self.bm["avgdl"]
        docs_tok = self.bm["doc_tokens"]
        ids = self.bm["ids"]
        qt = Counter(self._tokenize(q))
        scores = []
        for i, toks in enumerate(docs_tok):
            tf = Counter(toks)
            s = 0.0
            for t, _ in qt.items():
                if t in idf and t in tf:
                    f = tf[t]
                    s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / max(1, avgdl)))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [(ids[i], s) for s, i in scores[:top]]

    def _vector(self, q: str, top: int, company_filter: str | None = None) -> list[tuple[str, float]]:
        v = self._embed(self._query_text(q))
        sims = self.emb @ v
        idx = np.argsort(-sims)
        out = []
        for i in idx:
            cid = self.ids[i]
            if company_filter and self.by_id[cid]["company"] != company_filter:
                continue
            out.append((cid, float(sims[i])))
            if len(out) >= top:
                break
        return out

    def _bm25_filtered(self, q: str, top: int, company_filter: str) -> list[tuple[str, float]]:
        return [(cid, s) for cid, s in self._bm25(q, 400)
                if self.by_id[cid]["company"] == company_filter][:top]

    # ---------- 融合 ----------
    def retrieve(self, q: str, mode: str = "auto", top: int = FINAL_TOP) -> tuple[list[dict], dict]:
        fans = re.findall(r"(贵州茅台|五粮液|泸州老窖|山西汾酒|洋河股份|古井贡酒|今世缘|口子窖|迎驾贡酒|老白干酒|金种子酒)", q)
        if mode == "auto":
            cross = any(k in q for k in ("哪家", "哪些", "谁", "排名", "最高", "最低", "对比", "所有", "11 家", "11家", "普遍", "各家"))
            mode = "fanout" if (cross and len(set(fans)) < 2) else ("multi" if len(set(fans)) >= 2 else "global")
        detail = {"mode": mode, "companies_named": sorted(set(fans))}

        def merge(pairs_list: list[list[tuple[str, float]]]) -> list[str]:
            score: dict[str, float] = {}
            for pairs in pairs_list:
                for rank, (cid, _) in enumerate(pairs, 1):
                    score[cid] = score.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            order = sorted(score, key=lambda c: -score[c])
            # 同一张表最多保留 2 段
            kept, per_table, per_company = [], {}, {}
            for cid in order:
                d = self.by_id[cid]
                tkey = f"{d['sec_code']}_{d['report_period']}_p{d['pdf_page']}_{d.get('table_title') or ''}"
                if d["type"] == "table":
                    if per_table.get(tkey, 0) >= 2:
                        continue
                    per_table[tkey] = per_table.get(tkey, 0) + 1
                if mode == "fanout":
                    c = d["company"]
                    if per_company.get(c, 0) >= FANOUT_PER_COMPANY:
                        continue
                    per_company[c] = per_company.get(c, 0) + 1
                kept.append(cid)
            return kept

        if mode == "fanout":
            pairs = [self._bm25(q, BM25_TOP), self._vector(q, VEC_TOP)]
            for _code, name in COMPANIES:
                pairs.append(self._bm25_filtered(q, FANOUT_PER_COMPANY, name))
                pairs.append(self._vector(q, FANOUT_PER_COMPANY, company_filter=name))
            ordered = merge(pairs)
            detail["fanout_companies"] = len({self.by_id[c]["company"] for c in ordered})
        elif mode == "multi":
            pairs = [self._bm25(q, BM25_TOP), self._vector(q, VEC_TOP)]
            for name in set(fans):
                pairs.append(self._bm25_filtered(q, 6, name))
                pairs.append(self._vector(q, 6, company_filter=name))
            ordered = merge(pairs)
        else:
            pairs = [self._bm25(q, BM25_TOP), self._vector(q, VEC_TOP)]
            ordered = merge(pairs)

        bm_rank = {cid: i for i, (cid, _) in enumerate(self._bm25(q, BM25_TOP), 1)}
        vec_rank = {cid: i for i, (cid, _) in enumerate(self._vector(q, VEC_TOP), 1)}
        out = []
        for cid in ordered[:top]:
            d = dict(self.by_id[cid])
            d["bm25_rank"] = bm_rank.get(cid)
            d["vec_rank"] = vec_rank.get(cid)
            out.append(d)
        return out, detail


CROSS_HINT = ("哪家", "哪些", "谁", "排名", "最高", "最低", "对比", "所有", "普遍",
              "各家", "分别", "11 家", "11家", "全部")


def _looks_cross_company(q: str) -> bool:
    """没有点名公司、又在问"哪家/所有/对比"这类 → 判定为跨公司全景题。"""
    named = re.findall(r"(贵州茅台|五粮液|泸州老窖|山西汾酒|洋河股份|古井贡酒|今世缘|口子窖|迎驾贡酒|老白干酒|金种子酒)", q)
    return len(set(named)) < 2 and any(k in q for k in CROSS_HINT)


def build_context(blocks: list[dict]) -> str:
    out = []
    for i, b in enumerate(blocks, 1):
        loc = f"{b['company']}·{b['report_period']}{b['report_type']}·{b['section']}"
        if b.get("printed_page"):
            loc += f"·第{b['printed_page']}页"
        if b.get("table_title"):
            loc += f"·表：{b['table_title']}"
        out.append(f"[{i}] 【{loc}】\n{b['text']}")
    return "\n\n".join(out)


def answer(q: str, retr: Retriever, mode: str = "auto", top: int = FINAL_TOP,
           model: str = LLM_MODEL) -> dict:
    # fanout 题（跨公司全景）必须把 11 家都放进窗口：否则就是课件诊断过的
    # 「挑到了 11 家、窗口只放得下 4 家」。全局题仍按 top 截断。
    if mode == "fanout" or (mode == "auto" and _looks_cross_company(q)):
        mode = "fanout"
        top = max(top, len(COMPANIES) * FANOUT_PER_COMPANY)
    blocks, detail = retr.retrieve(q, mode=mode, top=top)
    if detail["mode"] == "fanout":
        detail["window"] = f"fanout 放开到 {len(blocks)} 块（11 家 × {FANOUT_PER_COMPANY}）"
    prompt = SYSTEM + STRICT_TAIL.replace("{context}", build_context(blocks)).replace("{q}", q)
    text = chat([{"role": "user", "content": prompt}], model=model, temperature=0, max_tokens=1200)

    # ---- 后置校验 ----
    cites = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)})
    bad_cites = [c for c in cites if c < 1 or c > len(blocks)]
    need, exempt = _num_tokens(text)
    joined = " ".join(b["text"] for b in blocks).replace(",", "")
    unsupported = sorted(n for n in need if n not in joined)
    return {
        "question": q, "answer": text, "mode": detail["mode"],
        "n_blocks": len(blocks),
        "citations": [{"n": c, "chunk_id": blocks[c - 1]["chunk_id"],
                       "company": blocks[c - 1]["company"],
                       "section": blocks[c - 1]["section"],
                       "printed_page": blocks[c - 1]["printed_page"],
                       "table_title": blocks[c - 1].get("table_title")}
                      for c in cites if 1 <= c <= len(blocks)],
        "invalid_citations": bad_cites,
        "number_check": {"checked": len(need), "exempt": sorted(exempt),
                         "unsupported": unsupported[:12], "ok": len(unsupported) == 0},
        "blocks": [{k: b.get(k) for k in ("chunk_id", "company", "section", "printed_page",
                                          "table_title", "type", "bm25_rank", "vec_rank",
                                          "index_text")} | {"text": b["text"][:800]}
                   for b in blocks],
        "detail": detail,
    }
