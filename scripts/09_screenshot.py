# -*- coding: utf-8 -*-
"""⑨ 给问答页面截图（作业 A 的交付物之一「页面截图」）。

用 selenium + 本机 chromedriver（crawler 环境已装 selenium）。
截图前先自己起好页面：  python scripts/08_app.py --port 7860

运行（crawler 环境）：
    python scripts/09_screenshot.py [--question "…"] [--port 7860]
产出：
    outputs/A_页面截图.png            首页 + 示例题按钮
    outputs/A_页面截图_回答.png        某道题的完整回答（含证据卡片）
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUT_DIR  # noqa: E402

CHROMEDRIVER = Path.home() / "chromedriver"


def new_driver() -> webdriver.Chrome:
    opt = Options()
    opt.add_argument("--headless=new")
    opt.add_argument("--window-size=1200,1500")
    opt.add_argument("--hide-scrollbars")
    service = Service(str(CHROMEDRIVER)) if CHROMEDRIVER.exists() else None
    return webdriver.Chrome(service=service, options=opt) if service else webdriver.Chrome(options=opt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--question", default="2026年上半年，11家白酒上市公司里营业收入最高和最低的分别是哪家？两者相差约多少倍？")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    d = new_driver()
    try:
        d.get(base)
        time.sleep(1.5)
        p1 = OUT_DIR / "A_页面截图.png"
        d.save_screenshot(str(p1))
        print(f"✅ {p1}")

        # 填问题并提交（POST 表单，不需要 JS）
        box = d.find_element(By.NAME, "q")
        box.clear()
        box.send_keys(args.question)
        d.find_element(By.CSS_SELECTOR, "button.go").click()
        time.sleep(75)                       # 等检索 + 模型作答
        d.execute_script("window.scrollTo(0, 0)")
        p2 = OUT_DIR / "A_页面截图_回答.png"
        d.save_screenshot(str(p2))
        print(f"✅ {p2}")
        body = d.find_element(By.TAG_NAME, "body").text
        print("页面文字长度：", len(body))
        print(body[:600])
    finally:
        d.quit()


if __name__ == "__main__":
    main()
