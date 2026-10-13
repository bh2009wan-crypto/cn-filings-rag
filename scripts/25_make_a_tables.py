# -*- coding: utf-8 -*-
"""㉕ 生成作业 A 的四张评价表：召回 / 答案 / 口径核对 / 全景取证。

把课件第三课 P61–P63、P90–P92 的三条口径落成可核验的表：

  · 「召回与答案表现分别评价」（P61 列各路线召回了什么块 / P63 列各路线写出了几家）
    → 表1 只放**检索层**指标，表2 只放**生成层**指标，两表不混列。
  · 「数字丢口径——是谁的、哪个口径、哪一期，要回到那张表」（P91）
    → 表3 把同一指标在**合并报表**与**母公司报表**两个出处并排列出。
  · 「全景按公司取证」（P61–P63：答案在十几家公司的共性与差异上）
    → 表4 对 11 家逐家定位「营业收入 + 同比」的原文位点，并标出本轮是否被召回。

只读 results_a.csv + data/chunks/chunks_a.jsonl + eval/key_facts.json。
**不调模型、不跑检索**，因此任何时候重跑都得到同一张表。

运行（crawler 环境）：
    python scripts/25_make_a_tables.py
产出（outputs/eval/，已在 .gitignore 白名单内）：
    表1_召回评价.md  表2_答案表现.md  表3_口径核对.md  表4_全景取证矩阵.md
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNK_DIR, EVAL_DIR, OUT_DIR  # noqa: E402

CSV_PATH = OUT_DIR / "eval" / "results_a.csv"
KFF = EVAL_DIR / "key_facts.json"
OUT = OUT_DIR / "eval"

# 11 家「营业收入 + 同比」的取证位点：chunk_id 是**逐家在语料里定位**出来的
# （第二节「主要会计数据」为主；洋河/泸州老窖/五粮液该行落在第三节「主要财务数据同比变动情况」）。
# 值不靠人记，每次运行都从被引用的块里**重新读出来并断言相等**——语料一变就报错。
REVENUE_EVIDENCE: list[tuple[str, str, str, float]] = [
    ("贵州茅台", "600519_2026H1_p005_t1", "90,703,260,964.48", 1.47),
    ("五粮液",   "000858_2026H1_p011_t1", "28,416,674,541.77", 20.87),
    ("泸州老窖", "000568_2026H1_p013_t1", "10,472,224,575.78", -36.35),
    ("山西汾酒", "600809_2026H1_p005_t1", "21,044,387,808.34", -12.18),
    ("洋河股份", "002304_2026H1_p012_t3", "10,540,431,576.04", -28.76),
    ("古井贡酒", "000596_2026H1_p007_t1", "10,131,395,927.11", -27.01),
    ("今世缘",   "603369_2026H1_p005_t3", "6,435,322,896.05", -7.41),
    ("口子窖",   "603589_2026H1_p005_t3", "1,951,744,051.30", -22.89),
    ("迎驾贡酒", "603198_2026H1_p005_t3", "3,415,678,280.00", 8.08),
    ("老白干酒", "600559_2026H1_p005_t3", "2,103,589,926.11", -15.23),
    ("金种子酒", "600199_2026H1_p005_t2", "236,013,679.60", -51.19),
]

# 表3 的口径对照行：(公司, 指标, 口径, 期, chunk_id, 值, 题目, 判定)
CALIBER_ROWS: list[tuple[str, str, str, str, str, str, str, str]] = [
    ("金种子酒", "营业收入", "合并报表", "2026H1", "600199_2026H1_p005_t2", "236,013,679.60",
     "Q9", "**正确值**；本轮没被召回，系统取了下面那行的母公司口径"),
    ("金种子酒", "营业收入", "母公司报表", "2026H1", "600199_2026H1_p037_t1", "189,805,515.53",
     "Q9", "**系统实际答的值**——「相差 477.9 倍」错在这一步（按合并口径应约 384 倍）"),
    ("贵州茅台", "归属于母公司股东的净利润", "合并报表", "2026H1", "600519_2026H1_p030_t1_3", "44,516,880,421.86",
     "Q8", "答对；系统同时给了下一行的母公司口径"),
    ("贵州茅台", "净利润", "母公司报表", "2026H1", "600519_2026H1_p032_t1_2", "17,062,381,326.67",
     "Q8", "答对；两者差 274 亿，是课件实验四的那个坑"),
    ("贵州茅台", "所有者权益合计", "合并报表", "2026-06-30", "600519_2026H1_p025_t1_5", "262,096,352,174.36",
     "Q6", "答对；该表**跨 p25–p28**，靠跨页合并才取全"),
]

NUM = re.compile(r"-?[\d,]+\.\d{2}%?")


def read_csv() -> list[dict]:
    return list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))


def read_chunks() -> dict[str, dict]:
    return {d["chunk_id"]: d
            for d in (json.loads(l) for l in
                      (CHUNK_DIR / "chunks_a.jsonl").read_text(encoding="utf-8").splitlines()
                      if l.strip())}


def rev_row(doc: dict) -> tuple[str, float]:
    """从被引用的块里读出「营业收入 | 本期 | 上期 | 同比」那一行。

    三种表形态都吃：`营业收入 | a | b | 1.47`（茅台）、
    `营业收入（元） | a | b | -27.01%`（古井贡酒）、
    `- | 营业收入 | - | a | - | - | b | - | - | 20.87% | -`（五粮液）。
    规则：该行里**第一个两位小数**是本报告期营业收入，**最后一个**是同比。
    """
    for line in doc["text"].splitlines():
        if "营业收入" not in line or "|" not in line:
            continue
        nums = NUM.findall(line)
        if len(nums) >= 2:
            return nums[0], float(nums[-1].rstrip("%"))
    raise AssertionError(f"{doc['chunk_id']} 里找不到「营业收入 + 同比」那一行")


def main() -> None:
    rows = read_csv()
    chunks = read_chunks()
    kfs = json.loads(KFF.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)

    scored = [r for r in rows if not r["type"].startswith("附加")]

    # ---------- 表1：召回评价（只用检索层指标）----------
    L1 = ["# 表1 · 召回评价（检索层）", "",
          "> 只看**检索**：gold 块有没有被召回、排第几。与「答得对不对」无关——",
          "> 那是表2 的事（课件 P61 看各路线召回了什么块，P63 才看各路线写出了几家）。",
          "> 生成：`python scripts/25_make_a_tables.py`　数据源：`outputs/eval/results_a.csv`", ""]
    L1 += ["| 题 | 题型 | 检索模式 | 召回块数 | gold 排名 | gold 进 top5 | gold 进 top10 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        n_blocks = r["retrieved_ids"].count(";") + 1 if r["retrieved_ids"] else 0
        L1.append(f"| {r['id']} | {r['type']} | {r['mode']} | {n_blocks} | "
                  f"{r['gold_rank'] or '—'} | {r['gold_in_top5'] or '—'} | {r['gold_in_top10'] or '—'} |")
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    def rate(rs, key):
        vals = [r[key] for r in rs if r[key] != ""]
        return f"{sum(1 for v in vals if v == '1')}/{len(vals)}" if vals else "—"

    L1 += ["", "## 按题型的召回命中率", "",
           "| 题型 | 题数 | gold 进 top5 | gold 进 top10 |", "|---|---|---|---|"]
    for t, rs in by_type.items():
        L1.append(f"| {t} | {len(rs)} | {rate(rs, 'gold_in_top5')} | {rate(rs, 'gold_in_top10')} |")
    all_gold = [r for r in rows if r["gold_in_top10"] != ""]
    L1 += ["",
           f"**合计：{sum(1 for r in all_gold if r['gold_in_top10'] == '1')}/{len(all_gold)} 题的 gold 块进了 top10。**",
           "",
           "两道跨公司全景题的 gold 都**掉出 top10**（Q9 第 18、Q10 第 19）——它们被召回了（进 22 块窗口），",
           "但在融合排序里靠后。召回覆盖不足是全景题失败的第一层原因，见 `表4_全景取证矩阵.md`。"]

    # ---------- 表2：答案表现（只用生成层指标）----------
    L2 = ["# 表2 · 答案表现（生成层）", "",
          "> 只看**作答**：事实命中、数字核对、角标合法性、人工判定。检索的好坏在表1。",
          "> `verdict`：2 对 / 1 部分对 / 0 错。`error_type`：E1 未召回·E2 召回但答错·E3 数字单位错·",
          "> E4 跨公司错位·E5 幻觉·E6 假性拒答·E7 该拒答却硬答（编码表见 `21_run_eval_a.py`）。", ""]
    L2 += ["| 题 | 题型 | 事实命中 | 数字核对 | 角标 | 人工判定 | 错误类型 | 判词 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L2.append(f"| {r['id']} | {r['type']} | {r['fact_hit'] or '—'} | {r['number_check']} | "
                  f"{r['citations']} | {r['verdict'] or '—'} | {r['error_type'] or '—'} | {r['note']} |")
    vc = Counter(r["verdict"] for r in scored if r["verdict"])
    vc_all = Counter(r["verdict"] for r in rows if r["verdict"])
    n_judged = sum(vc_all.values())          # 打了分的题数（含附加题）
    ok_num = sum(1 for r in rows if r["number_check"] == "ok")
    ok_cite = sum(1 for r in rows if r["invalid_citations"] == "0")
    facts = [r["fact_hit"] for r in scored if r["fact_hit"]]
    fh = sum(int(f.split("/")[0]) / max(1, int(f.split("/")[1])) for f in facts) / len(facts) if facts else 0
    score = sum(int(x) for x in vc_all.elements())
    L2 += ["", "## 汇总", "",
           f"- 计入评测 **{len(scored)} 题**：{vc.get('2', 0)} 题全对 / {vc.get('1', 0)} 题部分对 / "
           f"{vc.get('0', 0)} 题错",
           f"- 附加题 Q11（拒答）判对 → 2/1/0 计分 **{score}/{2 * n_judged}**"
           f"（与 `19_verdict_a.py` 同口径，分母含附加题）",
           f"- 事实命中率（按题取均值）：**{fh:.0%}**",
           f"- 数字核对通过：**{ok_num}/{len(rows)}**（含附加题）",
           f"- 角标全部合法（无引用不存在的编号）：**{ok_cite}/{len(rows)}**",
           f"- 错误类型分布：{dict(Counter(r['error_type'] for r in rows if r['error_type'])) or '无'}"]

    # ---------- 表3：口径与出处核对 ----------
    L3 = ["# 表3 · 口径与出处核对", "",
          "> 课件 P91：「数值进了图谱只剩一个节点……**是谁的、哪个口径、哪一期——要回到那张表**。」",
          "> 下表把同一个指标在**合并报表**与**母公司报表**两个出处并排列出；",
          "> 每个值都在脚本里从被引用的块中重新读出并断言相等（语料变了会报错）。", ""]
    L3 += ["| 公司 | 指标 | 口径 | 期 | 出处（章节 · 第几页 · 块ID） | 值 | 相关题 | 判定 |",
           "|---|---|---|---|---|---|---|---|"]
    for comp, item, cal, period, cid, val, qid, verdict in CALIBER_ROWS:
        doc = chunks.get(cid)
        assert doc is not None, f"块不存在：{cid}"
        assert val in doc["text"], f"{cid} 里没有 {val}"
        assert comp == doc["company"], f"{cid} 不是 {comp} 的块"
        L3.append(f"| {comp} | {item} | {cal} | {period} | "
                  f"{doc['section']} · 第 {doc['printed_page']} 页 · `{cid}` | {val} | {qid} | {verdict} |")
    L3 += ["",
           "**这张表要说明的一件事**：同一个「营业收入」在合并利润表和母公司利润表里是两个数。",
           "Q9 之所以给出「相差 477.9 倍」，不是取错了数，而是**取对了数、认错了表**——"
           "189,805,515.53 那种量级的数字只在母公司报表里出现。这也解释了为什么数字核对过了却仍然是错答案：",
           "**数字核对只能防编造，防不住口径错**（错的数字确实来自某块原文）。"]

    # ---------- 表4：全景取证矩阵 ----------
    recall_of = {"Q9": set(by_id["Q9"]["retrieved_ids"].split(";")),
                 "Q10": set(by_id["Q10"]["retrieved_ids"].split(";"))}
    L4 = ["# 表4 · 全景取证矩阵（按公司取证）", "",
          "> 课件 P61–P63：行业全景问题「没有一块能单独回答」，**答案在十几家公司之间的共性与差异上**，",
          "> 每条路线都只看到一角。所以全景题的正确做法是**逐家公司在各自材料里取证**，再汇总。",
          "> 下表对 11 家逐家定位「营业收入 + 同比」那一行的原文位点，" ,
          "> 并标出**本轮它有没有被检索到**——这正是 Q10「把没召回说成原文没有」的正面回应。", ""]
    L4 += ["| 公司 | 营业收入(元) | 同比 | 出处（章节 · 第几页 · 块ID） | Q9 是否召回 | Q10 是否召回 |",
           "|---|---|---|---|---|---|"]
    up, down, missed = [], [], []
    for comp, cid, want_rev, want_yoy in REVENUE_EVIDENCE:
        doc = chunks.get(cid)
        assert doc is not None, f"块不存在：{cid}"
        rev, yoy = rev_row(doc)
        assert rev == want_rev, f"{comp} 营业收入对不上：块里是 {rev}，表里写 {want_rev}"
        assert abs(yoy - want_yoy) < 1e-9, f"{comp} 同比对不上：块里是 {yoy}，表里写 {want_yoy}"
        mark = lambda q: "✅" if cid in recall_of[q] else "❌"
        L4.append(f"| {comp} | {rev} | {yoy:+.2f}% | {doc['section']} · 第 {doc['printed_page']} 页 · `{cid}` | "
                  f"{mark('Q9')} | {mark('Q10')} |")
        (up if yoy > 0 else down).append(comp)
        if cid not in recall_of["Q10"]:
            missed.append(comp)
    L4 += ["",
           f"- 11 家的真实分布：**同比上升 {len(up)} 家**（{'、'.join(up)}）；"
           f"**下降 {len(down)} 家**（{'、'.join(down)}）",
           f"- Q10 的回答说「其余 8 家原文未给出同比数据」。这 8 家的证据块**全部都在库里**"
           f"（见上表 Q10 列的 ❌ 共 {len(missed)} 处）——只是没进本轮的 22 块窗口。",
           "  所以那句话的准确说法是「**没检索到**」，不是「**原文没有**」；"
           "上升家数也不是回答里说的 2 家，而是 3 家（漏掉的正是迎驾贡酒）。",
           "",
           "- Q9 的取证缺口同理：金种子酒的**合并口径**营业收入 `236,013,679.60` 就在 "
           "`600199_2026H1_p005_t2`（第二节 第 5 页），Q9 的窗口里没有它，",
           "  系统改用了**母公司报表**的 `189,805,515.53`（见 `表3_口径核对.md`），于是得出了错的倍数。"]

    for name, lines in [("表1_召回评价.md", L1), ("表2_答案表现.md", L2),
                        ("表3_口径核对.md", L3), ("表4_全景取证矩阵.md", L4)]:
        p = OUT / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"→ {p}")

    print(f"\n校验通过：11 家取证位点逐个核对一致；三处口径对照的块与值都在语料里。")
    print(f"召回 gold 进 top10：{sum(1 for r in all_gold if r['gold_in_top10'] == '1')}/{len(all_gold)}"
          f"｜答案 2/1/0 计分：{score}/{2 * n_judged}")


if __name__ == "__main__":
    main()
