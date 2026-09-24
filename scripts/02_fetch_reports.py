# -*- coding: utf-8 -*-
"""② 从巨潮下载财报 PDF。

- 作业 A：白酒 11 家的 2026 年半年报（category_bndbg_szsh）
- 作业 B：贵州茅台 2021–2025 年报 + 2021H1–2026H1 半年报（2026 年报尚不存在）

运行（crawler 环境）：
    python scripts/02_fetch_reports.py --set A [--only 600519]
    python scripts/02_fetch_reports.py --set B
产出：
    data/pdf/<code>_<报告期>_<类型>.pdf
    data/pdf/manifest.json   （含 orgId/adjunctUrl/报告期/字节数，供后续步骤与可追溯）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (CATEGORY_ANNUAL, CATEGORY_HALF, CNINFO_QUERY, CNINFO_STATIC,  # noqa: E402
                    COMPANIES, COMPANIES_FILE, COMPANY_B, PDF_DIR,
                    PERIODS_B_ANNUAL, PERIODS_B_HALF, PERIOD_A, REQUEST_INTERVAL,
                    REPORTS_B_FILE, SEARCH_DATE_A, SEARCH_DATE_B)
from utils.http import CNINFO_HEADERS, download, post  # noqa: E402

MANIFEST = PDF_DIR / "manifest.json"
EXCLUDE_TITLE = ("摘要", "英文", "已取消", "意见", "通知", "更正公告的公告")


def load_companies() -> dict[str, dict]:
    if not COMPANIES_FILE.exists():
        raise SystemExit("❌ 先跑 scripts/01_resolve_orgids.py（缺 config/companies.json）")
    return {c["code"]: c for c in json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))}


def query_announcements(code: str, org_id: str, column: str, category: str,
                        se_date: str, page_size: int = 30) -> list[dict]:
    """查公告列表（自动翻页）。返回原始 announcement 字典列表。"""
    out, page = [], 1
    while True:
        d = post(CNINFO_QUERY, {
            "stock": f"{code},{org_id}", "tabName": "fulltext",
            "pageSize": page_size, "pageNum": page, "column": column,
            "category": category, "seDate": se_date, "isHLtitle": "true",
        })
        anns = d.get("announcements") or []
        out.extend(anns)
        total = d.get("totalRecordNum") or 0
        if len(out) >= total or not anns or page >= 5:
            break
        page += 1
        time.sleep(REQUEST_INTERVAL)
    return out


def pick_report(anns: list[dict], period: str, rep_type: str) -> dict | None:
    """在公告里挑出目标报告：按标题关键字匹配，剔除摘要/英文/取消；有更正版取最新。"""
    if rep_type == "年报":
        keys = (f"{period}年年度报告", f"{period} 年年度报告")
    else:
        year = period.replace("H1", "")
        keys = (f"{year}年半年度报告", f"{year} 年半年度报告")
    cands = []
    for a in anns:
        t = (a.get("announcementTitle") or "").replace(" ", "")
        if not any(k.replace(" ", "") in t for k in keys):
            continue
        if any(x in t for x in EXCLUDE_TITLE):
            continue
        cands.append(a)
    if not cands:
        return None
    # 有「更正」的按公告时间取最新（更正版即最终版）
    cands.sort(key=lambda a: a.get("announcementTime") or 0)
    return cands[-1]


def fetch_one(code: str, name: str, org_id: str, column: str, period: str,
              rep_type: str, se_date: str, manifest: dict) -> bool:
    category = CATEGORY_ANNUAL if rep_type == "年报" else CATEGORY_HALF
    key = f"{code}_{period}_{rep_type}"
    if key in manifest and (PDF_DIR / manifest[key]["file"]).exists():
        print(f"  ⏭  {name} {period} {rep_type}：已下载，跳过")
        return True
    anns = query_announcements(code, org_id, column, category, se_date)
    if not anns:
        print(f"  ❌ {name} {period} {rep_type}：公告列表返回 0 条（orgId 或 column 有问题，必须查）")
        return False
    a = pick_report(anns, period, rep_type)
    if not a:
        print(f"  ❌ {name} {period} {rep_type}：{len(anns)} 条公告里没匹配到该期报告")
        return False
    # 同一份公告可能在 A / B 两个清单里各出现一次（如 600519 的 2026 半年报）——
    # 按 adjunctUrl 去重，复用已下载的文件，避免同一份报告被切两次块
    dup = next((k for k, v in manifest.items() if v.get("adjunctUrl") == a["adjunctUrl"]
                and (PDF_DIR / v["file"]).exists()), None)
    if dup:
        src = manifest[dup]
        print(f"  ⏭  {name} {period} {rep_type}：与 {dup} 是同一份公告，复用其文件")
        manifest[key] = {**src, "period": period, "report_type": rep_type, "alias_of": dup}
        return True

    url = CNINFO_STATIC + a["adjunctUrl"]
    fname = re.sub(r"[^\w\-.]", "_", f"{key}__{a['announcementTitle']}")[:80] + ".pdf"
    expect = (a.get("adjunctSize") or 0) * 1024 or None
    download(url, PDF_DIR / fname, expect_bytes=expect, headers=CNINFO_HEADERS)
    manifest[key] = {
        "code": code, "name": name, "period": period, "report_type": rep_type,
        "title": a["announcementTitle"], "adjunctUrl": a["adjunctUrl"],
        "announcement_time": a.get("announcementTime"), "file": fname,
        "bytes": (PDF_DIR / fname).stat().st_size, "source": "cninfo(巨潮)",
    }
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["A", "B", "all"], default="A")
    ap.add_argument("--only", default=None, help="只处理某一家（代码，如 600519）")
    args = ap.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    companies = load_companies()
    ok = fail = 0

    if args.set in ("A", "all"):
        print(f"\n=== 作业 A：白酒 11 家 · {PERIOD_A} 半年报 ===")
        for code, name in COMPANIES:
            if args.only and code != args.only:
                continue
            c = companies.get(code)
            if not c:
                print(f"  ❌ {code} {name}：companies.json 里没有"); fail += 1; continue
            if fetch_one(code, name, c["orgId"], c["column"], PERIOD_A, "半年报",
                         SEARCH_DATE_A, manifest):
                ok += 1
            else:
                fail += 1
            time.sleep(REQUEST_INTERVAL)

    if args.set in ("B", "all"):
        code, name = COMPANY_B
        c = companies.get(code)
        if not c:
            raise SystemExit(f"❌ companies.json 里没有 {code}")
        print(f"\n=== 作业 B：{name} 2021–2026 年报 + 半年报 ===")
        for period in PERIODS_B_ANNUAL:
            if fetch_one(code, name, c["orgId"], c["column"], period, "年报",
                         SEARCH_DATE_B, manifest): ok += 1
            else: fail += 1
            time.sleep(REQUEST_INTERVAL)
        for period in PERIODS_B_HALF:
            if fetch_one(code, name, c["orgId"], c["column"], period,
                         "半年报", SEARCH_DATE_B, manifest): ok += 1
            else: fail += 1
            time.sleep(REQUEST_INTERVAL)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ manifest 共 {len(manifest)} 份（{MANIFEST}）｜本次成功 {ok}、失败 {fail}")
    if fail:
        raise SystemExit(f"❌ 有 {fail} 份没下成，先修再往下走")


if __name__ == "__main__":
    main()
