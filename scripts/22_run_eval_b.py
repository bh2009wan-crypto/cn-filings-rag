# -*- coding: utf-8 -*-
"""㉒ 把 Colab 回传的 results_b.json / results_b.csv 汇总成交付用的对照表。

运行（crawler 环境）：
    python scripts/22_run_eval_b.py [--src ~/Downloads/results_b.json]
产出：
    outputs/B_对照表.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUT_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads" / "results_b.json"))
    args = ap.parse_args()
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"❌ 找不到 {src}（Colab 跑完请把 results_b.json 下载回本机）")

    d = json.loads(src.read_text(encoding="utf-8"))
    summary = d.get("summary", {})

    grid: dict[str, dict[str, dict]] = defaultdict(dict)
    for k, v in summary.items():
        model, cell = k.split("|", 1)
        grid[cell][model] = v

    lines = ["# 作业 B · 四格对照（训练前 / 训练后）", "",
             f"来源：`{src.name}`（{d.get('n_rows','?')} 条逐题记录）｜基座：{d.get('base_model','?')}"
             f"｜设备：{d.get('device','?')}｜精度：{d.get('dtype','?')}"
             f"｜每格上限：{d.get('cap_per_cell','?')} 条",
             "",
             "| 格 | 题数 | **训练前·字符F1** | **训练后·字符F1** | 训练前·数字命中 | 训练后·数字命中 | 训练前·长度比 | 训练后·长度比 | 误拒答(前→后) |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = ["格1_训练过_原问法", "格2_训练过_换问法", "格3_没训练过_原问法",
             "格4_没训练过_换问法", "格5_RAG对照_测试集"]
    for cell in order:
        if cell not in grid:
            continue
        b, a = grid[cell].get("训练前", {}), grid[cell].get("训练后", {})
        lines.append(
            f"| {cell} | {(b or a).get('n','?')} "
            f"| {(b or {}).get('char_f1','-')} | {(a or {}).get('char_f1','-')} "
            f"| {(b or {}).get('num_hit','-')}（{(b or {}).get('n_with_numbers','-')} 题） "
            f"| {(a or {}).get('num_hit','-')}（{(a or {}).get('n_with_numbers','-')} 题） "
            f"| {(b or {}).get('len_ratio','-')} | {(a or {}).get('len_ratio','-')} "
            f"| {(b or {}).get('refused','-')} → {(a or {}).get('refused','-')} |")

    ti = d.get("train_info") or {}
    if ti:
        lines += ["", "**训练配置**：" +
                  f"步数 {ti.get('steps','?')}｜耗时 {ti.get('seconds','?')}s｜" +
                  f"可训练参数 {ti.get('trainable_params',0)/1e6:.2f}M｜" +
                  f"target_modules {','.join(ti.get('target_modules') or [])}"]

    lines += ["", "## 怎么读", "",
              "- **主指标是字符 bigram F1**（金答案里 68% 根本没有数字，只看数字命中会全是 0）。",
              "- **格1 应显著高于格3**：教过的事实能答，没教过的答不了——SFT 的「能与界」。",
              "- **格2 若明显低于格1**：说明它背的是问法，不是事实（课件实测：原题 100%、换问法 78%）。",
              "- **格5（RAG 对照）** = 同一基座 + 检索原文：若格5 不输格3，说明这类问题该建库而不是微调。",
              "- **长度比**接近 1 说明学会了「答多长」，显著大于 1 说明还在啰嗦。",
              "- 误拒答 = 原文里有、它却说「没有相关内容」。",
              "- ⚠️ 基座是 **Qwen3-0.6B**（本机 M2 能跑得动的最大选择），课件用的是 Qwen3.5-2B——容量差异会影响绝对分数。"]

    out = OUT_DIR / "B_对照表.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
