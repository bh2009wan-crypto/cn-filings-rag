#!/bin/bash
# 把仓库 homework3 改名为 cn-filings-rag（GitHub 改名 + 同步本机所有引用）
#
# 背景：2026-09-24 夜 GitHub 从这台机器不可达（代理能连 google 但连不上 github），
# 所以改名这件事写成一键脚本，网络恢复后跑一次即可。
#
# 用法：
#     bash ~/work/homework3/scripts/rename_repo.sh            # 预演（只检查，不改任何东西）
#     bash ~/work/homework3/scripts/rename_repo.sh --apply    # 真做
#
# 做的事：
#   1) 检查 GitHub 是否可达（直连 / 代理两条路都试）
#   2) 调 API 把仓库名改成 cn-filings-rag（若你已在网页改过，会自动跳过）
#   3) 更新本地 git remote、README/docs 里的仓库链接、记忆与数据技能地图
#   4) 提交并推送
set -u
# 注意：本脚本已在 2026-09-24 执行过一次（homework3 → cn-filings-rag）；
# 留着当"改名+同步引用"的模板，再改名时改 OLD/NEW 即可。
OLD="homework3"; NEW="cn-filings-rag"; WHO="bh2009wan-crypto"
APPLY=""; [ "${1:-}" = "--apply" ] && APPLY=1
PROJ="$HOME/work/homework3"

say() { printf '%s\n' "$*"; }
reach() { curl -s -o /dev/null -w "%{http_code}" --max-time 12 "$@" https://api.github.com 2>/dev/null; }

say "== 1/4 检查 GitHub 可达性 =="
D=$(reach --noproxy '*'); P=$(reach -x http://127.0.0.1:7897)
say "   api.github.com  直连=$D  代理=$P"
if [ "$D" = "200" ]; then MODE="--noproxy *";
elif [ "$P" = "200" ]; then MODE="-x http://127.0.0.1:7897";
else say "   ❌ GitHub 仍不可达——换个节点或稍后再试"; exit 2; fi

say "== 2/4 取凭据并改名 =="
umask 077
printf 'protocol=https\nhost=github.com\n\n' | git credential fill > /tmp/.rn 2>/dev/null || true
TOK=$(sed -n 's/^password=//p' /tmp/.rn | head -1); rm -f /tmp/.rn
[ -z "$TOK" ] && { say "   ❌ 钥匙串里没取到 GitHub 凭据"; exit 3; }

curl -s --max-time 25 $MODE -H "Authorization: token $TOK" \
  "https://api.github.com/repos/$WHO/$OLD" -o /tmp/.chk
if grep -q '"full_name"' /tmp/.chk && ! grep -q "\"name\": \"$OLD\"" /tmp/.chk; then
  say "   已是新名字，跳过改名（你大概在网页改过了）"
else
  if [ -z "$APPLY" ]; then say "   （预演）将要执行：PATCH /repos/$WHO/$OLD  {\"name\":\"$NEW\"}"; else
    code=$(curl -s -o /tmp/.rn2 -w "%{http_code}" --max-time 30 $MODE -X PATCH \
      "https://api.github.com/repos/$WHO/$OLD" \
      -H "Authorization: token $TOK" -H "Accept: application/vnd.github+json" \
      -d "{\"name\":\"$NEW\"}")
    say "   改名 HTTP=$code"
    [ "$code" != "200" ] && { say "   ❌ 改名失败，见 /tmp/.rn2"; exit 4; }
  fi
fi

say "== 3/4 同步本机引用（只改 GitHub 链接，不动 ~/work/homework3 这个目录名）=="
# 只替换 "bh2009wan-crypto/homework3" 这种链接形式；目录路径 ~/work/homework3 保持不动
# 用数组 + 逐项引号，避免中文路径被分词（第一版就是这么挂的）
targets=()
for g in "$PROJ/README.md" "$PROJ"/docs/*.md "$PROJ"/notebooks/*.ipynb \
         "$HOME/work/数据与技能地图.md" \
         "$HOME/.claude/projects/-Users-lionel/memory/homework-llm-course-3.md" \
         "$HOME/.claude/projects/-Users-lionel/memory/MEMORY.md"; do
  [ -e "$g" ] && targets+=("$g")
done
for f in "${targets[@]}"; do
  [ -f "$f" ] || continue
  n=$(grep -c "$WHO/$OLD" "$f" 2>/dev/null || echo 0)
  [ "$n" = "0" ] && continue
  say "   $f：$n 处"
  [ -n "$APPLY" ] && sed -i '' "s|$WHO/$OLD|$WHO/$NEW|g" "$f"
done

say "== 4/4 更新 remote 并推送 =="
cd "$PROJ" || exit 5
if [ -n "$APPLY" ]; then
  git remote set-url origin "https://github.com/$WHO/$NEW.git"
  say "   remote → $(git remote get-url origin)"
  git add -A && git -c user.name=$WHO -c user.email=bh2009wan@126.com \
    commit -q -m "改名同步：仓库 homework3 → $NEW（GitHub 会自动重定向旧链接）" || true
  git push -u origin main 2>&1 | tail -3
else
  say "   （预演）将执行：git remote set-url origin https://github.com/$WHO/$NEW.git && git push"
  say ""
  say "预演完成。真做请加 --apply"
fi
say ""
say "提示：改完别忘了给新仓库加 Topics（老仓库的 Topics 会保留）"
