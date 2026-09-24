# -*- coding: utf-8 -*-
"""③ PDF → 结构化页（文字 + 章节 + 表格）。

这一步是质量核心，四件事都做了（每一条都来自实测踩坑）：
  1. 页眉页脚剥离 + **页码偏移逐份探测**（实测：茅台偏移 0，古井贡酒偏移 +1）
  2. 章节双路探测（目录 + 正文）并交叉校验
  3. 表格按行列还原 + **跨页同构表合并**（实测：茅台合并资产负债表跨 p26–p29）+ 表题回溯
  4. **正文剔除表格区域**（实测：不剔时 p27 有 979 字，剔后 28 字——表格文字在正文里重复了一遍）

运行（crawler 环境）：
    python scripts/03_parse_pdf.py --set A [--only 600519]
产出：
    data/text/<code>_<报告期>.jsonl   每行一页 {pdf_page, printed_page, section, text, tables}
    data/text/parse_report.json       汇总（页数/字数/表数/offset/章节探测状态）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PDF_DIR, TEXT_DIR  # noqa: E402

MANIFEST = PDF_DIR / "manifest.json"
SUMMARY = TEXT_DIR / "parse_report.json"

# 页码正则（实测三种形态）：茅台的 1/110、古井的 ~ 141 ~、以及「第 N 页」
PAGE_PATTERNS = [
    re.compile(r"(\d+)\s*/\s*(\d+)\s*$"),
    re.compile(r"[~～]\s*(\d+)\s*[~～]\s*$"),
    re.compile(r"第\s*(\d+)\s*页\s*$"),
    re.compile(r"^(\d{1,3})$"),          # 五粮液/泸州老窖/洋河：页脚就是一个裸数字
]
SECTION_RE = re.compile(r"^\s*第\s*([一二三四五六七八九十]+)\s*节\s*([^\n]{0,60})")
TOC_LINE_RE = re.compile(r"第\s*([一二三四五六七八九十]+)\s*节\s*(.{2,40}?)[\s.·…]{4,}(\d{1,3})\s*$")
UNIT_LINE_RE = re.compile(r"^单位[:：]|^编制单位|币种|^注[:：]|^资料来源|^数据来源")
NUMERIC_LINE_RE = re.compile(r"^[\d\s,.\-—－%()（）]+$")
# 不是表题的行：勾选框行、日期/期间行、目录行、以冒号结尾的长引导句
CHECKBOX_LINE_RE = re.compile(r"^[√□✓✗]|适用\s*[□√]|^[（(]?[0-9一二三四五六七八九十]+[)）]?\s*$")
DATE_LINE_RE = re.compile(r"^[\d]{4}\s*年|^[0-9]{4}[-/年]")
TOC_LINE_RE_ANY = re.compile(r"[.·…]{4,}")


def cell_join(parts: list[str]) -> str:
    """同一单元格被换行拆开的多段拼回：CJK 直接拼，其余之间留一个空格。"""
    res = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not res:
            res = p
        elif re.search(r"[一-鿿（）、，。：；]$", res) or re.match(r"^[一-鿿（）、，。：；]", p):
            res += p
        else:
            res += " " + p
    return res


def clean_cell(v) -> str:
    if v is None:
        return ""
    return cell_join(str(v).replace("　", " ").splitlines())


def page_number_of(lines: list[str]) -> int | None:
    for ln in reversed([l for l in lines if l.strip()][-2:]):
        for pat in PAGE_PATTERNS:
            m = pat.search(ln.strip())
            if m:
                return int(m.group(1))
    return None


TITLE_KEYWORDS = ("资产负债表", "利润表", "现金流量表", "所有者权益变动表",
                  "情况", "状况", "明细", "表")


def table_title_above(page, bbox, header_like: set[str]) -> str:
    """表题回溯：表格上沿往上 90pt 区域里找带"表/情况/状况/明细"关键词的行。

    实测的坑：pdfplumber 有时把表的前一两行漏在 bbox 之外，那几行会变成"表上方的文字"，
    于是朴素做法会把「结算备付金」「向中央银行借款」这类**表内行**当成表题。
    所以这里只认带关键词的行；找不到就留空（空标题是诚实的，块标签另有"章节·页码"可用）。
    """
    top = max(0, bbox[1] - 90)
    if bbox[1] <= 2:
        return ""
    try:
        region = page.crop((0, top, page.width, bbox[1] - 1))
        txt = region.extract_text() or ""
    except Exception:
        return ""
    for ln in reversed([l.strip() for l in txt.splitlines() if l.strip()]):
        if ln in header_like or len(ln) < 4 or len(ln) > 40:
            continue
        if (UNIT_LINE_RE.match(ln) or NUMERIC_LINE_RE.match(ln) or CHECKBOX_LINE_RE.match(ln)
                or DATE_LINE_RE.match(ln) or TOC_LINE_RE_ANY.search(ln)
                or SECTION_RE.match(ln) or ln.endswith("：") or ln.endswith("。")):
            continue
        if any(k in ln for k in TITLE_KEYWORDS):
            return ln[:60]
    return ""


def looks_like_header(row: list[str]) -> bool:
    """判断一行是不是"表头行"：没有纯数字单元格，且含表头常见词。"""
    cells = [c.strip() for c in row if c.strip()]
    if len(cells) < 2:
        return False
    if any(re.fullmatch(r"[\d,.\-—()（）%]+", c) for c in cells):
        return False                       # 有纯数字 → 是数据行
    joined = " ".join(cells)
    return any(k in joined for k in ("项目", "附注", "期末", "期初", "本期", "上期", "上年",
                                     "金额", "余额", "比例", "名称", "单位", "序号", "合计"))


def parse_one(pdf_path: Path, code: str, name: str, period: str, rep_type: str) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        pages_raw = []
        for i, page in enumerate(pdf.pages, 1):
            raw_text = page.extract_text() or ""
            lines = [l for l in raw_text.splitlines() if l.strip()]
            try:
                found = page.find_tables()
            except Exception:
                found = []
            tables = []
            for t in found:
                rows = [r for r in ([clean_cell(c) for c in row] for row in t.extract())
                        if any(r)]
                if not (len(rows) >= 2 and len(rows[0]) >= 2):    # 至少 2 行 2 列才算表
                    continue
                inline_title = ""
                # 表内标题：首行是"跨整行的合并单元格"才算（用单元格几何判断，不用"只有一格有字"——
                # 实测"结算备付金"这类**表内行**也常常只有一格有字，会被误判成标题）
                try:
                    raw_cells = t.rows[0].cells
                    non_none = [(i, c) for i, c in enumerate(raw_cells) if c is not None]
                    tb_width = t.bbox[2] - t.bbox[0]
                    if (len(rows) > 3 and len(non_none) == 1 and tb_width > 0
                            and (non_none[0][1][2] - non_none[0][1][0]) > tb_width * 0.7
                            and rows[0][0].strip()):
                        inline_title = rows[0][0].strip()
                        rows = rows[1:]
                except Exception:
                    pass
                tables.append({"bbox": list(t.bbox), "rows": rows,
                               "title": "", "inline_title": inline_title})
            # 剔除表格区域内的字符 → 干净正文（避免表格文字在正文里重复一遍）
            if tables:
                def keep(obj):
                    x0, x1 = obj.get("x0"), obj.get("x1")
                    top, bottom = obj.get("top"), obj.get("bottom")
                    if None in (x0, x1, top, bottom):
                        return True
                    cx, cy = (x0 + x1) / 2, (top + bottom) / 2
                    for t in tables:
                        bx0, btop, bx1, bbot = t["bbox"]
                        if bx0 <= cx <= bx1 and btop <= cy <= bbot:
                            return False
                    return True
                try:
                    body = page.filter(keep).extract_text() or ""
                except Exception:
                    body = raw_text
            else:
                body = raw_text
            pages_raw.append({"pdf_page": i, "lines": lines, "body": body, "tables": tables,
                              "height": page.height, "width": page.width, "page_obj": page})

        # ---- ① 页眉/页脚（出现频次 > 50% 的行）与页码偏移 ----
        freq: Counter = Counter()
        for p in pages_raw:
            if p["lines"]:
                freq[p["lines"][0]] += 1
                freq[p["lines"][-1]] += 1
        n = len(pages_raw)
        header_like = {ln for ln, c in freq.items() if c > n * 0.5}

        printed = [(p["pdf_page"], page_number_of(p["lines"])) for p in pages_raw]
        offsets = Counter(pdf_pg - pr for pdf_pg, pr in printed if pr)
        offset, offset_share = None, 0.0
        if offsets:
            offset, cnt = offsets.most_common(1)[0]
            offset_share = cnt / max(1, sum(offsets.values()))
            if offset_share < 0.9:
                offset = None                     # 不稳定就不敢用，只用 pdf_page

        # ---- ② 章节：目录 + 正文双路 ----
        # 注意：同一页可能含两个节首（实测 p25 同时出现"第七节 债券相关情况"和"第八节 财务报告"），
        # 所以正文侧按"页内行号"全部记下来，供 04_chunk 精确切分；不能只记每页第一个。
        sections_body: list[tuple[int, int, str]] = []      # (pdf_page, 页内行号, 节标签)
        toc_entries = []
        for p in pages_raw:
            for idx, ln in enumerate(p["lines"]):
                if ln in header_like:
                    continue
                m = SECTION_RE.match(ln)
                if m and "…" not in ln and ln.count(".") <= 6:
                    title = m.group(2).strip()
                    label = f"第{m.group(1)}节 {title}" if title else f"第{m.group(1)}节"
                    sections_body.append((p["pdf_page"], idx, label))
                tm = TOC_LINE_RE.match(ln.strip())
                if tm:
                    toc_entries.append({"num": tm.group(1), "title": tm.group(2).strip(),
                                        "printed_page": int(tm.group(3))})
        section_starts = sections_body

        def section_of(pdf_page: int) -> str:
            """该页开头的章节 = 之前所有页里出现的最后一个节首。"""
            cur = "封面/目录"
            for pg, _idx, label in sections_body:
                if pg < pdf_page:
                    cur = label
                else:
                    break
            return cur

        # ---- ③ 表题回溯（在页对象还在时做） ----
        for p in pages_raw:
            for t in p["tables"]:
                t["title"] = (t.get("inline_title")
                              or table_title_above(p["page_obj"], t["bbox"], header_like))

        # ---- ③'' 每页的章节切分（供合并判断与 04_chunk 使用） ----
        carried = "封面/目录"
        for p in pages_raw:
            body_lines = [l for l in p["body"].splitlines()
                          if l.strip() and l.strip() not in header_like]
            while body_lines and any(pat.search(body_lines[-1].strip()) for pat in PAGE_PATTERNS):
                body_lines.pop()
            spans = []
            for i, ln in enumerate(body_lines):
                m = SECTION_RE.match(ln.strip())
                if m and "…" not in ln and ln.count(".") <= 6:
                    title = m.group(2).strip()
                    spans.append({"start": i,
                                  "label": f"第{m.group(1)}节 {title}" if title else f"第{m.group(1)}节"})
            if not spans or spans[0]["start"] > 0:
                spans = [{"start": 0, "label": carried}] + spans
            p["section_field"] = spans
            p["section"] = carried                 # 该页开头的章节
            p["body_lines"] = body_lines
            carried = spans[-1]["label"]

        # ---- ③' 跨页同构表合并 ----
        merged: dict[int, list[dict]] = {p["pdf_page"]: [] for p in pages_raw}
        page_by_no = {p["pdf_page"]: p for p in pages_raw}
        pending = None
        for p in pages_raw:
            for t in p["tables"]:
                t = dict(t)
                t["page_range"] = [p["pdf_page"], p["pdf_page"]]
                if pending is not None:
                    prev = pending
                    prev_page = page_by_no[prev["page_range"][1]]
                    # 保守合并：前表贴底、后表贴顶、页相邻、同一章节、列数一致，
                    # 且后表自己没有表题（有自己表题 = 是另一张表，不能并）
                    if ((not t["title"] or t["title"] == prev["title"])
                            and not looks_like_header(t["rows"][0])       # 后表自带表头 = 另一张表
                            and len(prev["rows"][0]) == len(t["rows"][0])
                            and prev["bbox"][3] > prev_page["height"] * 0.72
                            and t["bbox"][1] < p["height"] * 0.28
                            and p["pdf_page"] == prev["page_range"][1] + 1
                            and prev_page["section_field"][-1]["label"] == p["section_field"][-1]["label"]):
                        if prev["rows"] and t["rows"] and prev["rows"][0] == t["rows"][0]:
                            t["rows"] = t["rows"][1:]                        # 重复表头去掉
                        prev["rows"].extend(t["rows"])
                        prev["page_range"][1] = p["pdf_page"]
                        continue
                pending = t
                merged[p["pdf_page"]].append(t)

    # ---- ④ 输出 ----
    out_pages = []
    for p in pages_raw:
        out_pages.append({
            "pdf_page": p["pdf_page"],
            "printed_page": (p["pdf_page"] - offset) if offset is not None else None,
            "section": p["section"],
            "sections": p["section_field"],
            "text": "\n".join(p["body_lines"]),
            "tables": [{"title": t["title"], "rows": t["rows"], "page_range": t["page_range"]}
                       for t in merged[p["pdf_page"]]],
        })

    out_file = TEXT_DIR / f"{code}_{period}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for pg in out_pages:
            f.write(json.dumps(pg, ensure_ascii=False) + "\n")

    return {
        "code": code, "name": name, "period": period, "report_type": rep_type,
        "pdf": pdf_path.name, "pdf_pages": len(out_pages),
        "page_offset": offset, "offset_share": round(offset_share, 3),
        "chars": sum(len(p["text"]) for p in out_pages),
        "tables": sum(len(p["tables"]) for p in out_pages),
        "table_rows": sum(len(t["rows"]) for p in out_pages for t in p["tables"]),
        "sections_found": len(section_starts), "toc_entries": len(toc_entries),
        "section_detect": "ok" if section_starts else "failed",
        "text_file": out_file.name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B", "all"], default="A")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit("❌ 先跑 scripts/02_fetch_reports.py")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}

    for key, m in manifest.items():
        if m.get("alias_of"):                 # 同一份公告的别名（A/B 清单各一份），只解析一次
            continue
        if args.only and m["code"] != args.only:
            continue
        if args.set == "A" and m["report_type"] != "半年报":
            continue
        if args.set == "B" and m["code"] != "600519":
            continue
        pdf_path = PDF_DIR / m["file"]
        if not pdf_path.exists():
            print(f"⚠️  缺文件，跳过：{m['file']}")
            continue
        info = parse_one(pdf_path, m["code"], m["name"], m["period"], m["report_type"])
        summary[key] = info
        warn = "" if info["section_detect"] == "ok" else "  ⚠️ 章节探测失败"
        print(f"  {info['name']:6s} {info['period']:7s} {info['pdf_pages']:4d} 页 "
              f"offset={info['page_offset']}({info['offset_share']}) "
              f"正文 {info['chars']:>7,} 字 表 {info['tables']:>3} 张/{info['table_rows']:>5} 行 "
              f"节 {info['sections_found']}{warn}")

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 汇总写入 {SUMMARY}")


if __name__ == "__main__":
    main()
