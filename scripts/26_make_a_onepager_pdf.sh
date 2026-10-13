#!/usr/bin/env bash
# ㉖ 把 docs/作业3_A一页结论.md 打成**一页** PDF（作业 A 的交付物之一）。
#
# 链路：pandoc → 自带样式的 HTML → Chrome headless --print-to-pdf。
# 这台机器上没有 LaTeX，所以 pandoc 直接出 PDF 那条路走不通。
#
# 两条硬约束，脚本末尾都会自检、不通过就非零退出：
#   ① 必须正好一页（样式在 docs/onepager.css，不要往里加 page-break）；
#   ② PDF 的文字层必须能搜到中文 —— 中文字体只能用 STHeiti / Lantinghei SC /
#      Arial Unicode MS 这一栏；换成 PingFang SC 之类，页面看着一样但搜出来是乱码。
#
# 运行：bash scripts/26_make_a_onepager_pdf.sh
# 产出：docs/作业3_A一页结论.pdf
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/docs/作业3_A一页结论.md"
CSS="$ROOT/docs/onepager.css"
PDF="$ROOT/docs/作业3_A一页结论.pdf"
HTML="$(mktemp -t onepager.XXXXXX).html"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PY="${PY:-/Users/lionel/anaconda3/envs/crawler/bin/python}"

trap 'rm -f "$HTML"' EXIT

[ -f "$CHROME" ] || { echo "❌ 找不到 Chrome：$CHROME"; exit 1; }

# --embed-resources 会把 onepager.css 一起内联进 HTML，脱离目录也能渲染
pandoc "$MD" -f markdown+raw_html -t html5 --standalone \
       --metadata title="作业3 A 一页结论" \
       --css="$CSS" --embed-resources -o "$HTML"

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
          --virtual-time-budget=30000 --print-to-pdf="$PDF" "file://$HTML" 2>/dev/null

"$PY" - "$PDF" <<'PYEOF'
import sys
from pypdf import PdfReader

r = PdfReader(sys.argv[1])
text = "".join(p.extract_text() or "" for p in r.pages)
keys = ["金种子酒", "召回", "口径", "迎驾贡酒", "4,462"]
hit = [k for k in keys if k in text]
print(f"页数 {len(r.pages)}｜文字层命中 {hit}")
if len(r.pages) != 1:
    sys.exit(f"❌ 出了 {len(r.pages)} 页，不是一页——按 docs/onepager.css 顶部注释里的顺序砍内容")
if len(hit) != len(keys):
    sys.exit(f"❌ 文字层搜不到 {[k for k in keys if k not in text]}——字体栈踩到 ToUnicode 那个坑了")
print("✅ 一页，且文字层可搜")
PYEOF

echo "→ $PDF"
