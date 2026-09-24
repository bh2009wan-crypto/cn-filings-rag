# -*- coding: utf-8 -*-
"""④ 切块 + 打四标签（公司·章节·页码·块序号）。

两个关键设计（都来自课件/实测）：
  1. 切块 800 字、重合 120、尽量在段落边界收尾；**不跨页、不跨节**（跨了就标签失真）。
  2. **标签拼进 index_text**（被检索的那份文本），不只是元数据——
     否则 BM25 与向量都看不见"公司/报告期"，跨公司问题（"11 家里谁营收最高"）永远召不回候选。
     块正文另存一份纯文本 text，用于展示与引用核对。

运行（crawler 环境）：
    python scripts/04_chunk.py --set A|B
产出：
    data/chunks/chunks_a.jsonl / chunks_b.jsonl
    data/chunks/chunk_stats.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_DIR, CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR, TEXT_DIR  # noqa: E402

MANIFEST = PDF_DIR / "manifest.json"


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落切块；单段超长硬切；相邻块保留 overlap 字重合。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > size:                      # 超长单段先硬切
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:size])
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}" if cur else p
        else:
            if cur:
                out.append(cur)
            tail = cur[-overlap:] if cur else ""
            cur = f"{tail}\n{p}" if tail else p
    if cur:
        out.append(cur)
    return [c for c in (x.strip() for x in out) if len(c) > 30]


def table_to_text(title: str, rows: list[list[str]]) -> str:
    """表格 → 可读文本：表题 + 表头 + 每行"科目 值 值"（保留空列占位，避免错列）。"""
    lines = []
    if title:
        lines.append(f"【表】{title}")
    for r in rows:
        cells = [c if c else "-" for c in r]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def split_table_rows(rows: list[list[str]], head: str, header_row: str,
                     size: int = CHUNK_SIZE) -> list[str]:
    """超大表按行分段，每段都带上"表题 + 表头"，保证任何一段都能被读懂、也能被检索命中。"""
    out, cur = [], head + header_row
    for r in rows[1:]:
        line = " | ".join(c if c else "-" for c in r)
        if len(cur) + len(line) + 1 > size and len(cur) > len(head) + len(header_row):
            out.append(cur)
            cur = head + header_row + "\n" + line
        else:
            cur += "\n" + line
    if cur.strip():
        out.append(cur)
    return [x for x in out if len(x) > 40]


def page_label(m: dict, section: str, printed_page, kind: str = "正文") -> str:
    pg = printed_page if printed_page else "?"
    return f"【{m['name']}·{m['code']}·{m['period']}{m['report_type']}·{section}·第{pg}页·{kind}】"


def build(prefix: str, manifest: dict, out_file: Path, stats: dict) -> None:
    n_chunk = n_text = n_table = 0
    len_dist, comp_counter, sec_counter = [], Counter(), Counter()
    with out_file.open("w", encoding="utf-8") as f:
        for key, m in manifest.items():
            if m.get("alias_of"):             # 别名条目跳过，避免同一份报告被切两次块
                continue
            if not key.startswith(prefix):
                continue
            tf = TEXT_DIR / f"{m['code']}_{m['period']}.jsonl"
            if not tf.exists():
                print(f"  ⚠️  缺解析文件：{tf.name}（先跑 03）")
                continue
            pages = [json.loads(l) for l in tf.read_text(encoding="utf-8").splitlines() if l.strip()]
            for p in pages:
                # ---- 正文块：按页内章节 span 切，绝不跨节 ----
                lines = p["text"].split("\n")
                spans = p.get("sections") or [{"start": 0, "label": p.get("section", "未知")}]
                for si, sp in enumerate(spans):
                    start = sp["start"]
                    end = spans[si + 1]["start"] if si + 1 < len(spans) else len(lines)
                    seg = "\n".join(lines[start:end]).strip()
                    if len(seg) <= 30:
                        continue
                    section = sp["label"]
                    for ci, txt in enumerate(split_text(seg), 1):
                        cid = f"{m['code']}_{m['period']}_p{p['pdf_page']:03d}_s{si}_c{ci}"
                        rec = {
                            "chunk_id": cid, "company": m["name"], "sec_code": m["code"],
                            "report": m["title"], "report_period": m["period"],
                            "report_type": m["report_type"], "section": section,
                            "pdf_page": p["pdf_page"], "printed_page": p["printed_page"],
                            "chunk_index": ci, "type": "text", "table_title": None,
                            "text": txt,
                            "index_text": page_label(m, section, p["printed_page"]) + "\n" + txt,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_chunk += 1; n_text += 1; len_dist.append(len(txt)); sec_counter[section] += 1
                # ---- 表格块：一张表一块；超大表按行分段，每段重复"表题+表头" ----
                for ti, t in enumerate(p["tables"], 1):
                    body = table_to_text(t["title"], t["rows"])
                    if len(body) < 40:
                        continue
                    pr = t.get("page_range") or [p["pdf_page"], p["pdf_page"]]
                    head = f"【表】{t['title']}\n" if t["title"] else ""
                    header_row = " | ".join(c if c else "-" for c in t["rows"][0]) if t["rows"] else ""
                    for part, rows in enumerate(split_table_rows(t["rows"], head, header_row), 1):
                        cid = f"{m['code']}_{m['period']}_p{p['pdf_page']:03d}_t{ti}" + \
                              (f"_{part}" if part > 1 else "")
                        rec = {
                            "chunk_id": cid, "company": m["name"], "sec_code": m["code"],
                            "report": m["title"], "report_period": m["period"],
                            "report_type": m["report_type"], "section": p.get("section", "未知"),
                            "pdf_page": p["pdf_page"], "printed_page": p["printed_page"],
                            "page_range": pr, "chunk_index": ti, "type": "table",
                            "table_title": t["title"], "rows": len(t["rows"]), "part": part,
                            "text": rows,
                            "index_text": (f"【{m['name']}·{m['code']}·{m['period']}{m['report_type']}"
                                           f"·{p.get('section','')}·第{pr[0]}-{pr[1]}页·表】\n{rows}"),
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_chunk += 1; n_table += 1; len_dist.append(len(rows))
                        sec_counter[p.get("section", "")] += 1

    len_dist.sort()
    q = lambda v: len_dist[int(len(len_dist) * v)] if len_dist else 0
    stats[prefix] = {
        "chunks": n_chunk, "text_chunks": n_text, "table_chunks": n_table,
        "chars_p50": q(0.5), "chars_p90": q(0.9), "chars_p99": q(0.99), "chars_max": len_dist[-1] if len_dist else 0,
        "sections": dict(sec_counter.most_common(12)),
        "file": out_file.name,
    }
    print(f"  {prefix}: {n_chunk} 块（正文 {n_text} + 表 {n_table}）"
          f" 字数 p50={q(0.5)} p90={q(0.9)} p99={q(0.99)} max={stats[prefix]['chars_max']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B", "all"], default="A")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    if args.set in ("A", "all"):
        # A 用 2026H1 的 11 家
        build("", {k: v for k, v in manifest.items() if v["period"] == "2026H1"},
              CHUNK_DIR / "chunks_a.jsonl", stats)
    if args.set in ("B", "all"):
        build("600519_", {k: v for k, v in manifest.items() if v["code"] == "600519"},
              CHUNK_DIR / "chunks_b.jsonl", stats)
    (CHUNK_DIR / "chunk_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"→ 统计写入 {CHUNK_DIR / 'chunk_stats.json'}")


if __name__ == "__main__":
    main()
