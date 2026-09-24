# -*- coding: utf-8 -*-
"""⑩ 造数据：从茅台各期财报原文生成「投资者提问 → 董秘作答」问答对。

对齐课件做法（22 份政策 → 421 条问答，提问/作答/质检三角色同一个便宜模型）：
  · 一次调用出一个块的 3 组问答（省 3× 成本）
  · **强制模型逐字抄出"依据原文"**——这一步让"编造"变成可字符串校验的：
    质检时拿这句去原文里 `in` 一下，抄不出来就是编的（比让模型自评可靠得多）
  · 输出严格 JSON；`thinking` 关闭（实测输出 token 132 → 1）

运行（crawler 环境）：
    python scripts/10_gen_qa.py [--limit 20] [--workers 8]
产出：
    data/sft/qa_raw.jsonl   原始问答（含后面被质检淘汰的，留作审计）
    data/sft/gen_log.json   调用数/token/耗时/失败块
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_DIR, LLM_MODEL, SFT_DIR, TODAY  # noqa: E402
from utils.llm import chat_json  # noqa: E402

PROMPT = """今天是 2026年9月24日。你在为贵州茅台董事会秘书准备**问答训练数据**。

只依据下面这段财报原文出题作答，禁止使用任何外部知识或推测。

要求：
1. 出 3 个**不同角度**的投资者提问（分别偏：经营与销售数据 / 业务与渠道变动 / 财务口径与会计科目）。
2. 答案用「董秘口径」：中性、口语、第一人称，先给结论再给依据；不要营销腔，不要预测股价，不要投资建议。
3. 答案里的每个数字都必须来自下面的原文；数字要带单位。
4. 每条问答都要给一句「依据原文」——**从下面原文里逐字抄出来的那一句**（不许改写、不许拼接）。
5. 严格输出 JSON 数组，不要 markdown 代码块、不要解释：
[{"问题":"…","答案":"…","依据原文":"…"}, …]

【财报原文】（{company} {period} {section} 第{page}页）
{text}
"""


def gen_one(rec: dict, retries: int = 3) -> list[dict]:
    prompt = (PROMPT.replace("{company}", rec["company"]).replace("{period}", rec["report_period"])
                    .replace("{section}", rec["section"]).replace("{page}", str(rec["printed_page"]))
                    .replace("{text}", rec["text"][:3000]))
    for attempt in range(1, retries + 1):
        try:
            data = chat_json([{"role": "user", "content": prompt}], model=LLM_MODEL,
                             temperature=0.7, max_tokens=2000)
            items = data if isinstance(data, list) else data.get("问答") or data.get("items") or []
            out = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                q = str(it.get("问题") or it.get("question") or "").strip()
                a = str(it.get("答案") or it.get("answer") or "").strip()
                g = str(it.get("依据原文") or it.get("evidence") or "").strip()
                if q and a:
                    out.append({"问题": q, "答案": a, "依据原文": g})
            if out:
                return out
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(1.5 * attempt + random.random())
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个块（试跑用）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-doc", type=int, default=0, help="每份报告最多取多少块（0=不限）")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in
              (CHUNK_DIR / "chunks_b.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 只用正文块；每份报告均匀抽样，避免被同一期报告占满
    text_chunks = [c for c in chunks if c["type"] == "text" and len(c["text"]) >= 200]
    by_doc: dict[str, list[dict]] = {}
    for c in text_chunks:
        by_doc.setdefault(c["report_period"] + c["report_type"], []).append(c)
    picked: list[dict] = []
    rnd = random.Random(20260924)
    for k, v in sorted(by_doc.items()):
        rnd.shuffle(v)
        need = args.per_doc or (len(text_chunks) // max(1, len(by_doc)))
        picked.extend(v[:need])
    if args.limit:
        picked = picked[:args.limit]
    print(f"  候选块 {len(text_chunks)}（{len(by_doc)} 份报告）→ 本次处理 {len(picked)} 块")

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = SFT_DIR / "qa_raw.jsonl"
    done: set[str] = set()
    if raw_file.exists():
        for l in raw_file.read_text(encoding="utf-8").splitlines():
            if l.strip():
                try:
                    done.add(json.loads(l)["chunk_id"])
                except Exception:
                    pass
        print(f"  ⏭  已完成 {len(done)} 块，跳过")

    todo = [c for c in picked if c["chunk_id"] not in done]
    lock = threading.Lock()
    n_pairs = n_fail = 0
    t0 = time.time()
    f = raw_file.open("a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(gen_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                pairs = fut.result()
            except Exception as e:
                n_fail += 1
                pairs = []
                print(f"    ❌ {c['chunk_id']}（{type(e).__name__}: {str(e)[:50]}）")
            with lock:
                for p in pairs:
                    f.write(json.dumps({
                        "chunk_id": c["chunk_id"], "company": c["company"],
                        "report_period": c["report_period"], "report_type": c["report_type"],
                        "section": c["section"], "printed_page": c["printed_page"],
                        "page_range": c.get("page_range"), "chunk_text": c["text"],
                        "问题": p["问题"], "答案": p["答案"], "依据原文": p["依据原文"],
                    }, ensure_ascii=False) + "\n")
                n_pairs += len(pairs)
                f.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"    {i}/{len(todo)} 块，{n_pairs} 条问答，{(i/max(el,1e-6)):.2f} 块/秒，"
                      f"已用 {el/60:.1f} 分，预计还需 {(len(todo)-i)/max(i/max(el,1e-6),1e-6)/60:.1f} 分")
    f.close()

    log = {"blocks_processed": len(todo) + len(done), "this_run": len(todo),
           "pairs_this_run": n_pairs, "failed_blocks": n_fail,
           "seconds": round(time.time() - t0, 1), "workers": args.workers,
           "model": LLM_MODEL, "seed": 20260924}
    (SFT_DIR / "gen_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n✅ 本次生成 {n_pairs} 条；累计写入 {raw_file}")


if __name__ == "__main__":
    main()
