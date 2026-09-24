# -*- coding: utf-8 -*-
"""⑭ 生成"换问法"版本（四格的第 2、4 格要用）。

只改问题的问法，**金标准答案不变**；改写后用 3-gram Jaccard 过滤掉"其实没换"的
（要求与原问相似度 < 0.5，跟课件一样：换装的问法才叫举一反三）。

运行（crawler 环境）：
    python scripts/14_make_paraphrase.py [--limit 30]
产出：
    data/sft/train_para.jsonl   data/sft/test_para.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LLM_MODEL, SFT_DIR  # noqa: E402
from utils.llm import chat  # noqa: E402

PROMPT = """把下面这个投资者的问题换一种问法，要求：
1. 意思完全不变，答案不受影响；可以更口语、更简短、或换个切入角度。
2. 不要出现"根据上述""材料中"这类元话语。
3. 只输出改写后的问题本身，不要引号、不要解释。

原问题：{q}
改写："""


def norm(s: str) -> str:
    return re.sub(r"[\s,，。、；：？！]", "", s or "")


def jaccard(a: str, b: str) -> float:
    ga = {a[i:i + 3] for i in range(max(0, len(a) - 2))}
    gb = {b[i:i + 3] for i in range(max(0, len(b) - 2))}
    return len(ga & gb) / max(1, len(ga | gb))


def rewrite(q: str) -> str:
    for attempt in range(1, 4):
        try:
            out = chat([{"role": "user", "content": PROMPT.replace("{q}", q)}],
                       model=LLM_MODEL, temperature=0.8, max_tokens=200).strip()
            out = out.strip("“”\"'　 \n")
            if 6 <= len(out) <= 80 and jaccard(norm(q), norm(out)) < 0.5:
                return out
        except Exception:
            time.sleep(1.2 * attempt)
    return ""


def process(src: Path, dst: Path, workers: int = 8) -> None:
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if dst.exists():
        done = {json.loads(l)["chunk_id"] + json.loads(l)["问题"]
                for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()}
    else:
        done = set()
    todo = [r for r in rows if r["chunk_id"] + r["问题"] not in done]
    print(f"  {src.name}: {len(rows)} 条，待改写 {len(todo)}")
    n_ok = n_skip = 0
    with dst.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(rewrite, r["问题"]): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = futs[fut]
            new_q = fut.result()
            if new_q:
                r = dict(r); r["原问题"] = r["问题"]; r["问题"] = new_q
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_ok += 1
            else:
                n_skip += 1        # 改写失败或"其实没换" → 不进这一格（宁缺毋滥）
            if i % 40 == 0 or i == len(todo):
                print(f"    {i}/{len(todo)}，成功 {n_ok}，丢弃 {n_skip}")
    print(f"  → {dst.name}：新增 {n_ok} 条（丢弃 {n_skip} 条相似度过高/失败的）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    for name in ("test", "train"):
        src = SFT_DIR / f"{name}.jsonl"
        if not src.exists():
            print(f"  ⚠️  缺 {src.name}（先跑 13_split.py）")
            continue
        if args.limit:
            rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()][:args.limit]
            tmp = SFT_DIR / f"{name}_subset.jsonl"
            tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            process(tmp, SFT_DIR / f"{name}_para.jsonl")
        else:
            process(src, SFT_DIR / f"{name}_para.jsonl")


if __name__ == "__main__":
    main()
