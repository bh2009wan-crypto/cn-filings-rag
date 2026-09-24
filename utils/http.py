# -*- coding: utf-8 -*-
"""网络请求助手：固定请求头 + 列表退避重试（沿用 daily-report/fetch_fund.py 的模式）。

运行：
    from utils.http import get, post, download
产出：无
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
MAX_RETRY = 3
BACKOFF = [2, 5, 10]          # 每次失败后等几秒再来

CNINFO_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
}


def _sleep(attempt: int) -> None:
    time.sleep(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)])


def post(url: str, data: dict | None = None, headers: dict | None = None,
         timeout: int = 30, retry: int = MAX_RETRY) -> dict:
    """POST 表单并返回 JSON。连续失败抛 RuntimeError（不静默）。"""
    last: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            r = requests.post(url, data=data, headers=headers or CNINFO_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:                      # 网络抖动/限流/超时都走这里
            last = e
            if attempt < retry:
                wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                print(f"    ⚠️  第 {attempt} 次 POST 失败（{type(e).__name__}: {e}），等 {wait} 秒重试…")
                time.sleep(wait)
    raise RuntimeError(f"POST {url} 连续 {retry} 次失败：{last}")


def get(url: str, headers: dict | None = None, timeout: int = 30,
        retry: int = MAX_RETRY) -> requests.Response:
    last: Exception | None = None
    for attempt in range(1, retry + 1):
        try:
            r = requests.get(url, headers=headers or {"User-Agent": UA}, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            if attempt < retry:
                wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                print(f"    ⚠️  第 {attempt} 次 GET 失败（{type(e).__name__}: {e}），等 {wait} 秒重试…")
                time.sleep(wait)
    raise RuntimeError(f"GET {url} 连续 {retry} 次失败：{last}")


def download(url: str, dest: Path, expect_bytes: int | None = None,
             headers: dict | None = None) -> Path:
    """下载到 dest。已存在且大小合理则跳过（幂等）。

    expect_bytes：巨潮返回的 adjunctSize（单位 KB）换算来的期望字节数，±5% 内视为已下好。
    """
    if dest.exists() and dest.stat().st_size > 1024:
        if expect_bytes is None or abs(dest.stat().st_size - expect_bytes) <= expect_bytes * 0.05:
            print(f"    ⏭  已存在，跳过：{dest.name}（{dest.stat().st_size/1e6:.2f} MB）")
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, headers=headers)
    blob = r.content
    if not blob.startswith(b"%PDF"):
        raise RuntimeError(f"❌ 下载到的不是 PDF（前 16 字节：{blob[:16]!r}）：{url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(dest)
    print(f"    ✅ 下载 {dest.name}（{len(blob)/1e6:.2f} MB）")
    return dest
