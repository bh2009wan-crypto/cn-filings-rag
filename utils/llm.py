# -*- coding: utf-8 -*-
"""DeepSeek 调用（OpenAI 兼容 HTTP，不装 SDK）。

运行：
    from utils.llm import chat, chat_json
    chat([{"role":"user","content":"…"}])                      # 返回纯文本
    chat_json([...], schema_hint="{...}")                      # 返回解析好的 dict/list

关键实测结论（写死在代码里，别再踩）：
  - 用 model="deepseek-flash"（便宜、非推理型）
  - 默认带 thinking 块，且会吃光 max_tokens；**一律传 thinking={"type":"disabled"}**
  - 走 OpenAI 兼容端点 {base}/chat/completions（base 取自 .env）
产出：无
"""
from __future__ import annotations

import json
import random
import time

import requests

from utils.envload import deepseek_cfg

MAX_RETRY = 4
BACKOFF = [2, 5, 10, 20]


def chat(messages: list[dict], model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 2048, json_mode: bool = False, thinking: bool = False,
         timeout: int = 180) -> str:
    """调一次模型，返回文本（自动跳过 thinking 块）。失败重试 MAX_RETRY 次。"""
    cfg = deepseek_cfg()
    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not thinking:
        payload["thinking"] = {"type": "disabled"}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last: Exception | None = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.post(f"{cfg['base']}/chat/completions",
                              headers={"Authorization": f"Bearer {cfg['key']}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=timeout)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            d = r.json()
            content = d["choices"][0]["message"].get("content")
            if isinstance(content, list):        # 兼容 content 为分块列表的形态
                content = "".join(p.get("text", "") for p in content
                                  if p.get("type") in (None, "text"))
            if not content:
                raise RuntimeError("模型返回空内容")
            return content
        except Exception as e:
            last = e
            if attempt < MAX_RETRY:
                wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)] + random.uniform(0, 1)
                print(f"    ⚠️  模型调用第 {attempt} 次失败（{type(e).__name__}: {str(e)[:80]}），等 {wait:.1f} 秒…")
                time.sleep(wait)
    raise RuntimeError(f"模型调用连续 {MAX_RETRY} 次失败：{last}")


def chat_json(messages: list[dict], **kw) -> object:
    """要求模型返回 JSON 并解析（json_mode 自动打开，容错剥掉围栏）。"""
    kw.setdefault("max_tokens", 4096)
    text = chat(messages, json_mode=True, **kw).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=-1)
        if start >= 0:
            end = max(text.rfind("]"), text.rfind("}"))
            return json.loads(text[start:end + 1])
        raise
