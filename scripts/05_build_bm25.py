# -*- coding: utf-8 -*-
"""⑤ 建 BM25 倒排索引（jieba 分词 + 自实现 BM25，不装 rank_bm25）。

对齐课件：BM25 是「1994 年的公式，至今仍是搜索引擎的底座」，与向量并排跑一条。
jieba 缺失时自动退化为字符二元组（零依赖兜底）。

运行（nlp 环境，因为装了 jieba）：
    python scripts/05_build_bm25.py --set A
产出：
    data/index/bm25_a.pkl
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_DIR, INDEX_DIR  # noqa: E402

K1, B = 1.5, 0.75
# 金融文本里高频但几乎不区分的词（自己列，不引外部停用词表）
STOP = set("的 和 及 与 或 在 是 为 了 元 万元 亿元 年 月 日 公司 报告 本期 上期 期末 期初 "
           "期末余额 期初余额 人民币 币种 单位 合计 其中 项目 附注 一 二 三 四 五 六 七 八 九 十 "
           "以及 及其 该 本 上述 如下 情况 说明 无 有 不 非".split())


def get_tokenizer():
    try:
        import jieba
        jieba.initialize()
        return (lambda s: [w for w in jieba.lcut(s) if w.strip() and w not in STOP and len(w) > 1]), "jieba"
    except Exception:
        def char2gram(s):
            s = "".join(ch for ch in s if not ch.isspace())
            return [s[i:i + 2] for i in range(len(s) - 1)]
        return char2gram, "char2gram"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B"], default="A")
    args = ap.parse_args()

    chunk_file = CHUNK_DIR / f"chunks_{args.set.lower()}.jsonl"
    if not chunk_file.exists():
        raise SystemExit(f"❌ 缺 {chunk_file}（先跑 04_chunk.py）")
    docs = [json.loads(l) for l in chunk_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    tok, tok_name = get_tokenizer()

    ids, doc_tokens, df = [], [], Counter()
    for d in docs:
        toks = tok(d["index_text"])
        ids.append(d["chunk_id"])
        doc_tokens.append(toks)
        for t in set(toks):
            df[t] += 1

    N = len(docs)
    avgdl = sum(len(t) for t in doc_tokens) / max(1, N)
    idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    out = INDEX_DIR / f"bm25_{args.set.lower()}.pkl"
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps({"ids": ids, "doc_tokens": doc_tokens, "idf": idf,
                                  "avgdl": avgdl, "k1": K1, "b": B, "tokenizer": tok_name},
                                 protocol=4))
    print(f"✅ BM25 建好：{N} 篇、词表 {len(df)}、avgdl {avgdl:.0f}、分词器 {tok_name}")
    print(f"→ {out}")

    # 自测探针：肉眼确认能召回对的公司
    print("\n=== 自测探针 ===")
    for q in ["贵州茅台 营业收入 同比", "毛利率 最高", "合同负债 经销商"]:
        qt = Counter(tok(q))
        scores = []
        for i, toks in enumerate(doc_tokens):
            tf = Counter(toks)
            s = 0.0
            for t, _ in qt.items():
                if t not in idf or t not in tf:
                    continue
                f = tf[t]
                s += idf[t] * f * (K1 + 1) / (f + K1 * (1 - B + B * len(toks) / max(1, avgdl)))
            if s:
                scores.append((s, i))
        scores.sort(reverse=True)
        top = [f"{scores[k][0]:.1f}·{docs[scores[k][1]]['company']}·{docs[scores[k][1]]['chunk_id'][-18:]}"
               for k in range(min(3, len(scores)))]
        print(f"  「{q}」→ " + ("｜".join(top) if top else "（无命中）"))


if __name__ == "__main__":
    main()
