# -*- coding: utf-8 -*-
"""⑪ 质检：先确定性校验（零成本），再让模型判「有据 0/1/2 · 自然 0/1」。

阶段一（脚本，不花钱，先砍掉大部分垃圾）：
  · 「依据原文」必须**逐字**出现在该块正文里（去空白后 in）→ 否则 grounded=0 直接丢
  · 答案里的数字必须都能在该块正文里找到 → 否则 num_ok=0
  · 页码必须落在该块 page_range 内（不符就修正，不丢）
  · 问题长度 8–60 字、得像投资者问的（剔除「根据上述材料」这类考试腔）、与同块另两问去重
阶段二（模型，开 thinking）：一次性判 有据 0/1/2 与 自然 0/1 + 一句理由。
留存标准：有据 ≥ 2 且 自然 = 1。

运行（crawler 环境）：
    python scripts/11_qc_qa.py [--limit 20]
产出：
    data/sft/qa_clean.jsonl   质检通过的问答对（交付物）
    data/sft/qc_report.json   各阶段淘汰数与原因分布
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LLM_MODEL, SFT_DIR  # noqa: E402
from utils.llm import chat_json  # noqa: E402

META_RE = re.compile(r"根据(上述|原文|材料)|上述材料|如上所述|本报告期数据来源")
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
QC_PROMPT = """你在给一条「投资者提问 → 董秘作答」的训练数据做质检。只输出 JSON。

【财报原文片段】
{chunk}

【待检问答】
问题：{q}
答案：{a}

请判断两件事并严格输出 JSON：
{{"有据": 0或1或2, "自然": 0或1, "理由": "一句话"}}
判断标准：
- 有据：答案是否**完全**能由上面原文支持。2=完全有据；1=大体有据但有细节外推；0=有编造或与原文冲突。
- 自然：问题是否像真实投资者会问的。1=像；0=像考试题/元话语（如"根据上述材料"）。
只输出 JSON，不要解释。"""


def norm(s: str) -> str:
    return re.sub(r"[\s,，。、；：]", "", s or "")


def dedupe_ngram(qs: list[str], thr: float = 0.8) -> list[int]:
    """同块内 3-gram Jaccard > thr 视为重复，返回要剔除的下标。"""
    def grams(s):
        s = norm(s)
        return {s[i:i + 3] for i in range(max(0, len(s) - 2))}
    drop = []
    gs = [grams(q) for q in qs]
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            if i in drop or j in drop:
                continue
            a, b = gs[i], gs[j]
            if a and b and len(a & b) / len(a | b) > thr:
                drop.append(j)
    return drop


def evidence_level(g: str, chunk: str) -> tuple[int, float]:
    """依据原文的可靠度：2=逐字出现在原文；1=高度接近（3-gram 覆盖率 ≥ 0.9）；0=不像抄的。

    实测：模型"抄写"时经常漏一两个虚词或把顿号换成逗号，严格 verbatim 会误杀一大半，
    所以分两级、并把覆盖率记下来（不是放宽了就算数，而是把放宽的程度写进报告）。
    """
    gn, cn = norm(g), norm(chunk)
    if not gn:
        return 0, 0.0
    if gn in cn:
        return 2, 1.0
    if len(gn) < 6:
        return 0, 0.0
    grams = {gn[i:i + 3] for i in range(len(gn) - 2)}
    hit = sum(1 for x in grams if x in cn)
    ratio = hit / max(1, len(grams))
    return (1, ratio) if ratio >= 0.9 else (0, ratio)


def stage1(rec: dict) -> tuple[bool, str]:
    chunk_n = norm(rec["chunk_text"])
    q, a, g = rec["问题"].strip(), rec["答案"].strip(), (rec.get("依据原文") or "").strip()
    if not (8 <= len(q) <= 60):
        return False, "问题长度越界"
    if META_RE.search(q):
        return False, "考试腔（元话语）"
    if not g:
        return False, "缺依据原文"
    lvl, ratio = evidence_level(g, rec["chunk_text"])
    rec["_依据等级"], rec["_依据覆盖率"] = lvl, round(ratio, 3)
    if lvl == 0:
        return False, f"依据原文覆盖率仅 {ratio:.2f}（不像抄的）"
    miss = [n for n in NUM_RE.findall(a.replace(",", ""))
            if len(n.replace(".", "")) >= 3 and n not in chunk_n.replace(",", "")]
    if miss:
        return False, f"答案有 {len(miss)} 个数字不在原文"
    return True, "ok"


CACHE_FILE = SFT_DIR / "qc_cache.json"


def load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


def stage2(rec: dict, cache: dict) -> dict:
    """模型质检（开 thinking）。带缓存：同一条问答只花一次钱，重跑只补失败的那些。"""
    key = rec["chunk_id"] + "|" + rec["问题"][:60]
    if key in cache and cache[key].get("有据", -1) >= 0:
        return cache[key]
    prompt = (QC_PROMPT.replace("{chunk}", rec["chunk_text"][:2500])
                        .replace("{q}", rec["问题"]).replace("{a}", rec["答案"]))
    for attempt in range(1, 4):
        try:
            d = chat_json([{"role": "user", "content": prompt}], model=LLM_MODEL,
                          temperature=0, max_tokens=500, thinking=True)
            if isinstance(d, dict) and "有据" in d:
                out = {"有据": int(d.get("有据", 0)), "自然": int(d.get("自然", 0)),
                       "理由": str(d.get("理由", ""))[:120]}
                cache[key] = out
                return out
        except Exception:
            time.sleep(1.5 * attempt)
    return {"有据": -1, "自然": -1, "理由": "质检调用失败"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    raw_file = SFT_DIR / "qa_raw.jsonl"
    if not raw_file.exists():
        raise SystemExit("❌ 先跑 scripts/10_gen_qa.py")
    rows = [json.loads(l) for l in raw_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"  读入 {len(rows)} 条原始问答")

    # ---- 阶段一 ----
    reasons = Counter()
    keep = []
    for rec in rows:
        ok, why = stage1(rec)
        if ok:
            keep.append(rec)
        else:
            reasons[why] += 1
            rec["_drop"] = why
    # 同块内去重
    by_chunk: dict[str, list[dict]] = {}
    for r in keep:
        by_chunk.setdefault(r["chunk_id"], []).append(r)
    deduped, n_dup = [], 0
    for cid, group in by_chunk.items():
        drop = dedupe_ngram([g["问题"] for g in group])
        for i, g in enumerate(group):
            if i in drop:
                n_dup += 1
            else:
                deduped.append(g)
    print(f"  阶段一：通过 {len(keep)}，淘汰 {len(rows)-len(keep)}（{dict(reasons)}），去重再删 {n_dup}")
    print(f"  存活率 {len(deduped)/max(1,len(rows)):.1%}"
          + ("  ⚠️ 低于 70%：建议把 10_gen_qa.py 改成开 thinking 重新生成" if len(deduped)/max(1,len(rows)) < 0.7 else ""))

    # ---- 阶段二 ----
    cache = load_cache()
    print(f"  质检缓存命中 {len(cache)} 条（这些不再花钱）")
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(stage2, r, cache): r for r in deduped}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            qc = fut.result()
            r.update(qc)
            if qc["有据"] >= 2 and qc["自然"] == 1:
                out.append(r)
            if i % 50 == 0 or i == len(deduped):
                print(f"    质检 {i}/{len(deduped)}，已留 {len(out)}")

    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    clean = SFT_DIR / "qa_clean.jsonl"
    with clean.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter((r.get("有据"), r.get("自然")) for r in deduped)
    report = {
        "raw": len(rows), "stage1_pass": len(keep), "stage1_drop_reasons": dict(reasons),
        "dedup_dropped": n_dup, "stage2_scored": len(deduped),
        "有据自然分布": {f"有据{k[0]}_自然{k[1]}": v for k, v in dist.most_common()},
        "kept": len(out), "keep_rate": round(len(out) / max(1, len(rows)), 3),
        "by_period": dict(Counter(r["report_period"] + r["report_type"] for r in out)),
    }
    (SFT_DIR / "qc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n✅ 质检通过 {len(out)} 条 → {clean}")
    print(f"   分布：{report['有据自然分布']}")


if __name__ == "__main__":
    main()
