# -*- coding: utf-8 -*-
"""⓪ 开工前体检：任一硬条件不满足就明确报错退出，不带着坏环境白跑一晚上。

检查项：
  · 两个解释器（crawler / nlp）可 import 关键库
  · torch 能不能用 MPS（本机是 M2；用不了就退 CPU，慢 3 倍）
  · 巨潮可达、hf-mirror 可达
  · .env 里的模型接入配置齐全，并**真调一次 1-token 的 ping**
  · 本地 embedding 模型是否已下载（缺就给出下载命令，不是让你猜）

运行（crawler 环境即可）：
    python scripts/00_env_check.py
产出：屏幕表格 + logs/00_env_check_<日期>.log
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import LOG_DIR, ROOT  # noqa: E402
from utils.envload import deepseek_cfg, load_env  # noqa: E402
from utils.http import get  # noqa: E402

CR = Path("/Users/lionel/anaconda3/envs/crawler/bin/python")
NL = Path("/Users/lionel/anaconda3/envs/nlp/bin/python")
MODEL_DIR = ROOT / "models" / "Qwen3-Embedding-0.6B"
EXPECT_MODEL_BYTES = 1191586416

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f"　{detail}" if detail else ""))


def py_has(python: Path, mods: list[str]) -> tuple[bool, str]:
    code = ("import importlib\n"
            "miss=[]\n"
            f"for m in {mods!r}:\n"
            "    try: importlib.import_module(m)\n"
            "    except Exception: miss.append(m)\n"
            "print(','.join(miss))")
    try:
        out = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=90)
        miss = out.stdout.strip()
        return (not miss), (f"缺 {miss}" if miss else "")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    load_env()
    print("== 解释器与依赖 ==")
    check("crawler 解释器存在", CR.exists(), str(CR))
    check("nlp 解释器存在", NL.exists(), str(NL))
    if CR.exists():
        ok, d = py_has(CR, ["requests", "pdfplumber", "pypdf", "bs4", "pandas"])
        check("crawler 关键库", ok, d)
    if NL.exists():
        ok, d = py_has(NL, ["torch", "transformers", "jieba", "flask"])
        check("nlp 关键库（torch/transformers/jieba/flask）", ok, d)

    print("\n== 算力 ==")
    if NL.exists():
        out = subprocess.run([str(NL), "-c",
                              "import torch;print(torch.backends.mps.is_available())"],
                             capture_output=True, text=True, timeout=120)
        mps = out.stdout.strip() == "True"
        check("MPS 可用（M2 上比 CPU 快约 3 倍）", mps, "" if mps else "将退 CPU，慢 3 倍")

    print("\n== 网络 ==")
    for name, url in (("巨潮资讯网", "https://www.cninfo.com.cn/"),
                      ("hf-mirror", "https://hf-mirror.com/api/models/Qwen/Qwen3-Embedding-0.6B"),
                      ("ModelScope", "https://modelscope.cn/api/v1/models/Qwen/Qwen3-Embedding-0.6B")):
        try:
            r = get(url, timeout=20, retry=2)
            check(name, True, f"HTTP {r.status_code}")
        except Exception as e:
            check(name, False, f"{type(e).__name__}")

    print("\n== 模型接入（.env）==")
    try:
        cfg = deepseek_cfg()
        check("配置齐全", True, f"base={cfg['base']} model={cfg['model']}")
        from utils.llm import chat
        txt = chat([{"role": "user", "content": "只回答两个字：可用"}], max_tokens=20)
        check("真调一次模型", bool(txt.strip()), f"回复：{txt.strip()[:20]}")
    except SystemExit as e:
        check("配置齐全", False, str(e))
    except Exception as e:
        check("真调一次模型", False, f"{type(e).__name__}: {str(e)[:60]}")

    print("\n== 本地 embedding 模型 ==")
    f = MODEL_DIR / "model.safetensors"
    if f.exists() and f.stat().st_size == EXPECT_MODEL_BYTES:
        check("Qwen3-Embedding-0.6B 权重", True, f"{f.stat().st_size/1e6:.0f} MB（字节数吻合）")
    else:
        size = f.stat().st_size if f.exists() else 0
        check("Qwen3-Embedding-0.6B 权重", False,
              f"当前 {size/1e6:.0f} MB / 期望 {EXPECT_MODEL_BYTES/1e6:.0f} MB")
        print("     → 下载办法（ModelScope 8 路并行，约十几分钟）："
              "\n       bash scripts/download_model.sh")

    print("\n== 阶段产物 ==")
    for name, path in (("语料清单 config/companies.json", ROOT / "config" / "companies.json"),
                       ("PDF manifest", ROOT / "data" / "pdf" / "manifest.json"),
                       ("切块 chunks_a.jsonl", ROOT / "data" / "chunks" / "chunks_a.jsonl"),
                       ("BM25 索引 bm25_a.pkl", ROOT / "data" / "index" / "bm25_a.pkl")):
        check(name, path.exists())
    # 向量要查"是否算完"：条数必须与切块数一致（不然只是跑了半截）
    ck = ROOT / "data" / "chunks" / "chunks_a.jsonl"
    ids = ROOT / "data" / "index" / "emb_a.ids.json"
    if ck.exists() and ids.exists():
        n_chunk = sum(1 for l in ck.read_text(encoding="utf-8").splitlines() if l.strip())
        n_vec = len(json.loads(ids.read_text(encoding="utf-8")))
        check("向量 emb_a.npy 是否算完", n_vec >= n_chunk, f"{n_vec}/{n_chunk} 条")
    else:
        check("向量 emb_a.npy 是否算完", False, "还没开始或没落盘")

    log = LOG_DIR / f"00_env_check_{datetime.now():%Y%m%d}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(f"{'OK ' if ok else 'FAIL'} {n} {d}" for n, ok, d in RESULTS),
                   encoding="utf-8")
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n→ {log}")
    if bad:
        raise SystemExit(f"❌ {len(bad)} 项未通过：{bad}")
    print("✅ 体检全过")


if __name__ == "__main__":
    main()
