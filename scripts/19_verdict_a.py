# -*- coding: utf-8 -*-
"""把人工判分（我逐条看答案后填）写进 results_a.csv 的 verdict / error_type / note 三列。

单独成脚本而不是一次性命令：判分理由里有很多中文引号与括号，写在文件里更可靠，
而且这份判分是可复核的（谁都能看到我给每道题打了几分、为什么）。

运行（crawler 环境）：
    python scripts/19_verdict_a.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUT_DIR  # noqa: E402

CSV_PATH = OUT_DIR / "eval" / "results_a.csv"

# verdict：2=对，1=部分对，0=错；error_type 见 21_run_eval_a.py 的编码表
VERDICT: dict[str, tuple[str, str, str]] = {
    "Q1": ("2", "", "营收 907.03 亿与 +1.47% 都对；并主动区分了「营业收入」与「营业总收入」两个口径"),
    "Q2": ("2", "", "五粮液归母净利 87.53 亿元，数字与口径都对"),
    "Q3": ("2", "", "汾酒营收 210.44 亿、归母净利 64.39 亿，两个数都对"),
    "Q4": ("2", "", "期末 31.78 亿、期初 80.07 亿、-60.31%，并说明是销售模式与预收货款政策调整"),
    "Q5": ("2", "", "经销商合计 4,740 家；分区域明细也对"),
    "Q6": ("2", "", "合并口径 2,620.96 亿正确——跨页表合并（p25–p28）起了作用"),
    "Q7": ("2", "", "41.44% 计算正确；明确标注合并报表口径"),
    "Q8": ("2", "", "445.17 亿（合并归母）与 170.62 亿（母公司）两个口径都对，差值也点出来了"),
    "Q9": ("1", "E3", "8 家营收枚举正确、最高是茅台正确；但最低那家用了错的口径——"
                       "金种子酒取的是母公司利润表的 189,805,515.53 元，合并营业收入是 236,013,679.60 元，"
                       "于是「相差 477.9 倍」这个结论错（应为约 384 倍）。另有 3 家（洋河/口子窖/迎驾贡酒）未召回"),
    "Q10": ("1", "E1;E6", "茅台 +1.47%、五粮液 +20.87%、汾酒 -12.18% 三家对；"
                          "但 8 家里的迎驾贡酒其实是 +8.08%（上升），它没被召回，"
                          "回答却写成「其余 8 家原文未给出同比数据」——把「没检索到」说成了「原文没有」"),
    "Q11": ("2", "", "正确拒答：半年报不披露研发人员数量，回答「检索到的原文中没有相关内容」，没有编造"),
}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    for r in rows:
        if r["id"] in VERDICT:
            r["verdict"], r["error_type"], r["note"] = VERDICT[r["id"]]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    v = {r["id"]: r["verdict"] for r in rows}
    print("判分写入完成：", v)
    print("得分：", sum(int(x) for x in v.values() if x), "/", 2 * len([x for x in v.values() if x]))


if __name__ == "__main__":
    main()
