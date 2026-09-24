#!/bin/bash
# 诊断"Colab 连不上"到底是代理的问题还是浏览器的问题
#
# 用法（在终端里敲）：
#     bash ~/work/homework3/scripts/check_colab_net.sh
#
# 判读：
#   · 代理检测全绿 → 代理没问题，是浏览器侧（Cookie/插件/多账号）或节点刚抖过
#   · 有红色       → 节点掉线或规则缺域名，换节点/改规则
set -u
PROXY="http://127.0.0.1:7897"

echo "=============================================="
echo " 1) 系统代理是否开着（浏览器走不走代理看这里）"
echo "=============================================="
scutil --proxy | grep -E "HTTPEnable|HTTPSEnable|SOCKSEnable|HTTPPort" | sed 's/^/   /'
echo "   期望：Enable 都为 1、Port 为 7897"

echo
echo "=============================================="
echo " 2) 通过代理访问 Colab 依赖的域名"
echo "=============================================="
check() {
  local name="$1" url="$2" expect="$3"
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -x "$PROXY" "$url" 2>/dev/null)
  if [ "$code" = "$expect" ] || { [ "$expect" = "2xx" ] && [ "${code:0:1}" = "2" ]; }; then
    printf "   ✅ %-46s %s\n" "$name" "$code"
  else
    printf "   ❌ %-46s %s（期望 %s）\n" "$name" "${code:-超时}" "$expect"
  fi
}
check "colab 主站"            "https://colab.research.google.com/"          "2xx"
check "gstatic（报错里那个）"  "https://ssl.gstatic.com/generate_204"        "204"
check "gstatic 主域"           "https://www.gstatic.com/generate_204"        "204"
check "googleusercontent"      "https://lh3.googleusercontent.com/generate_204" "204"
check "googleapis"             "https://www.googleapis.com/generate_204"     "204"
check "google 主域"            "https://www.google.com/generate_204"         "204"

echo
echo "=============================================="
echo " 3) 直连能不能通（应该不通，通说明没走代理）"
echo "=============================================="
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 --noproxy '*' https://colab.research.google.com/ 2>/dev/null)
if [ "$code" = "000" ]; then echo "   ✅ 直连不通（000）—— 符合预期，必须走代理"
else echo "   ⚠️  直连返回 $code —— 你可能没开代理，浏览器会走直连然后失败"; fi

echo
echo "=============================================="
echo " 4) WebSocket 升级握手（Colab 靠它保持连接）"
echo "=============================================="
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -x "$PROXY" \
  -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "https://colab.research.google.com/" 2>/dev/null)
if [ "$code" = "200" ] || [ "$code" = "101" ]; then
  echo "   ✅ 握手 $code —— 代理支持长连接"
else
  echo "   ❌ 握手 ${code:-超时} —— 这个节点对长连接不友好，换节点"
fi

echo
echo "=============================================="
echo " 判读：全绿 = 代理没问题，看浏览器（Cookie/插件/多账号）"
echo "       有红 = 换节点；只有 3) 是 ⚠️ = 去 Clash 里打开「系统代理」"
echo " 顺带：浏览器里打开 https://ssl.gstatic.com/generate_204"
echo "       能显示空白页 = 浏览器侧通；打不开 = 浏览器没走代理或被拦"
echo "=============================================="
