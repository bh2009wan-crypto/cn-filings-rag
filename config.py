# -*- coding: utf-8 -*-
"""全局配置：公司清单、路径、切块与检索参数。

诚实说明：orgId 不能从股票代码推出来（实测 11 家里 4 家猜错、静默返回 0 条），
所以只保留代码与简称，orgId 一律由 scripts/01_resolve_orgids.py 现场解析落盘。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- 路径 ----------
DATA = ROOT / "data"
PDF_DIR = DATA / "pdf"
TEXT_DIR = DATA / "text"
CHUNK_DIR = DATA / "chunks"
INDEX_DIR = DATA / "index"
SFT_DIR = DATA / "sft"
CONFIG_DIR = ROOT / "config"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "outputs"
EVAL_DIR = ROOT / "eval"

COMPANIES_FILE = CONFIG_DIR / "companies.json"      # 01 脚本产出（入库）
REPORTS_B_FILE = CONFIG_DIR / "reports_b.json"      # B 的报告清单（入库）

# ---------- 作业 A：白酒 11 家 2026 年半年报 ----------
COMPANIES: list[tuple[str, str]] = [
    ("600519", "贵州茅台"),
    ("000858", "五粮液"),
    ("000568", "泸州老窖"),
    ("600809", "山西汾酒"),
    ("002304", "洋河股份"),
    ("000596", "古井贡酒"),
    ("603369", "今世缘"),
    ("603589", "口子窖"),
    ("603198", "迎驾贡酒"),
    ("600559", "老白干酒"),
    ("600199", "金种子酒"),
]
PERIOD_A = "2026H1"
REPORT_TYPE_A = "半年报"
SEARCH_DATE_A = "2026-06-01~2026-09-30"

# ---------- 作业 B：贵州茅台 2021–2026 年报 + 半年报 ----------
COMPANY_B = ("600519", "贵州茅台")
# 2026 年报尚不存在（约 2027-04 才披露），README 里已写明
PERIODS_B_ANNUAL = ["2021", "2022", "2023", "2024", "2025"]
PERIODS_B_HALF = ["2021H1", "2022H1", "2023H1", "2024H1", "2025H1", "2026H1"]
SEARCH_DATE_B = "2021-01-01~2026-09-24"

# ---------- 巨潮接口 ----------
CNINFO_SEARCH = "https://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "https://static.cninfo.com.cn/"
CATEGORY_HALF = "category_bndbg_szsh"     # 半年报
CATEGORY_ANNUAL = "category_ndbg_szsh"    # 年报
REQUEST_INTERVAL = 1.5                    # 每次请求之间礼貌等 1.5 秒

# ---------- 切块（对齐课件：800 字 / 重合 120） ----------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---------- 检索 ----------
VEC_TOP = 20          # 向量召回条数
BM25_TOP = 20         # BM25 召回条数
RRF_K = 60            # RRF 融合常数
FINAL_TOP = 8         # 进 prompt 的块数（跨公司全景题 fanout 时为 22）
FANOUT_PER_COMPANY = 2

# ---------- embedding ----------
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMB_DIM = 1024
EMB_MAX_LEN = 768     # 由 04_chunk.py 的 token 分位数定，可在 06 脚本里用 --max-len 覆盖
HF_ENDPOINT = "https://hf-mirror.com"

# ---------- 模型（A 作答 / B 造数据） ----------
LLM_MODEL = "deepseek-flash"
LLM_MODEL_STRONG = "deepseek-v4-pro"
TODAY = "2026年9月24日"   # 必须注入：模型内部日期是 2026-05-07，会假性拒答 2026 半年报
