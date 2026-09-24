# -*- coding: utf-8 -*-
"""① 解析 11 家白酒公司的巨潮 orgId。

为什么单独一步：**orgId 不能从股票代码推**——实测按 gssh0/gssz0+代码猜，
洋河/今世缘/口子窖/迎驾贡酒 4 家会静默返回 0 条公告（最危险的失败方式：看起来像"这家没发财报"）。
所以必须先调 topSearch 拿真值并落盘。

运行（crawler 环境）：
    cd ~/work/homework3 && /Users/lionel/anaconda3/envs/crawler/bin/python scripts/01_resolve_orgids.py [--force]
产出：
    config/companies.json   [{code, name, orgId, column}, ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import COMPANIES, COMPANIES_FILE, CNINFO_SEARCH, REQUEST_INTERVAL  # noqa: E402
from utils.http import post  # noqa: E402


def column_of(code: str) -> str:
    """6 开头走上交所，其余走深交所。"""
    return "sse" if code.startswith("6") else "szse"


def main() -> None:
    force = "--force" in sys.argv
    if COMPANIES_FILE.exists() and not force:
        old = json.loads(COMPANIES_FILE.read_text(encoding="utf-8"))
        if len(old) == len(COMPANIES) and all(o.get("orgId") for o in old):
            print(f"⏭  {COMPANIES_FILE.name} 已有 {len(old)} 家且都带 orgId，跳过（--force 可重解析）")
            return

    out, failed = [], []
    for code, name in COMPANIES:
        try:
            d = post(CNINFO_SEARCH, {"keyWord": name, "maxNum": 10})
            hit = next((x for x in d if x.get("code") == code), None)
            if not hit:
                failed.append({"code": code, "name": name, "reason": f"topSearch 无匹配（返回 {len(d)} 条）"})
                print(f"❌ {code} {name}：topSearch 里找不到该代码")
            else:
                out.append({"code": code, "name": name, "orgId": hit["orgId"],
                            "column": column_of(code), "category": hit.get("category")})
                print(f"✅ {code} {name:6s} orgId={hit['orgId']} column={column_of(code)}")
        except Exception as e:
            failed.append({"code": code, "name": name, "reason": f"{type(e).__name__}: {e}"})
            print(f"❌ {code} {name}：{type(e).__name__}: {e}")
        time.sleep(REQUEST_INTERVAL)

    COMPANIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMPANIES_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 写入 {COMPANIES_FILE}（{len(out)} 家）")

    if failed:
        log = COMPANIES_FILE.parent.parent / "logs" / "01_failed.json"
        log.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"❌ {len(failed)} 家解析失败，详见 {log}（orgId 错会静默返回 0 条，必须修）")
    print("✅ 11 家全部解析成功")


if __name__ == "__main__":
    main()
