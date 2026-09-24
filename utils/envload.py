# -*- coding: utf-8 -*-
"""环境变量加载器（沿用 ~/work/daily-report/mailer.py 的写法）。

运行：
    from utils.envload import load_env
    load_env()            # 读项目根目录的 .env，setdefault 语义（系统环境变量优先）

产出：无（只改 os.environ）
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """把 .env 里的键值塞进 os.environ（已存在的不覆盖）。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def deepseek_cfg() -> dict:
    """返回调用 DeepSeek 所需的三项配置；缺任何一项都明确报错。"""
    load_env()
    base = os.environ.get("DEEPSEEK_BASE_URL")
    key = os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    # 兼容 daily-report 的键名（ANTHROPIC_BASE_URL 指向 api.deepseek.com/anthropic）
    if not base or not key:
        ab = os.environ.get("ANTHROPIC_BASE_URL", "")
        base = base or ab.replace("/anthropic", "")
        key = key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    if not base or not key:
        raise SystemExit("❌ .env 里缺少模型接入配置（DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY）")
    return {"base": base.rstrip("/"), "key": key, "model": model}
