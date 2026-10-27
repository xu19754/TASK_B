#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf '同步停止：%s\n' "$1" >&2
  exit 1
}

root=$(git rev-parse --show-toplevel 2>/dev/null) || fail '请在原项目 Git 工作区内运行。'
cd "$root"

[[ $(git branch --show-current) == main ]] || fail '请先切换到原项目的 main 分支。'
[[ -z $(git status --porcelain --untracked-files=normal) ]] || fail '工作区有未提交的文件；请先检查并提交或妥善保留它们。'

# 固定两个远程仓库，避免在团队仓库的 main 上误运行时推送到团队 main。
check_remote() {
  local name=$1 expected=$2 url found=0
  while IFS= read -r url; do
    found=1
    case "$url" in
      "https://github.com/$expected"|"https://github.com/$expected.git"|\
      "git@github.com:$expected"|"git@github.com:$expected.git"|\
      "ssh://git@github.com/$expected"|"ssh://git@github.com/$expected.git") ;;
      *) fail "远程 $name 的地址不是预期仓库：$url" ;;
    esac
  done < <(git config --get-all "remote.$name.url" || true)
  [[ $found == 1 ]] || fail "未配置远程 $name。"
  while IFS= read -r url; do
    case "$url" in
      "https://github.com/$expected"|"https://github.com/$expected.git"|\
      "git@github.com:$expected"|"git@github.com:$expected.git"|\
      "ssh://git@github.com/$expected"|"ssh://git@github.com/$expected.git") ;;
      *) fail "远程 $name 的推送地址不是预期仓库：$url" ;;
    esac
  done < <(git config --get-all "remote.$name.pushurl" || true)
}

check_remote origin yxh0527-ship-it/taskb
check_remote placement xu19754/TASK_B

source_ref=refs/heads/main
target_ref=refs/heads/ARM_to-placement_YE
git fetch --no-tags origin "$source_ref:refs/remotes/origin/main"
git fetch --no-tags placement "$target_ref:refs/remotes/placement/ARM_to-placement_YE"

source_commit=$(git rev-parse "$source_ref")
origin_commit=$(git rev-parse refs/remotes/origin/main)
target_commit=$(git rev-parse refs/remotes/placement/ARM_to-placement_YE)
git merge-base --is-ancestor "$origin_commit" "$source_commit" || \
  fail '原仓库 main 有本地未包含的提交；请先同步并检查，不能强制推送。'

if git merge-base --is-ancestor "$source_commit" "$target_commit"; then
  result_commit=$target_commit
elif git merge-base --is-ancestor "$target_commit" "$source_commit"; then
  result_commit=$source_commit
else
  # 先在 Git 对象数据库中做三方合并，不切换或修改当前工作区。
  if ! merge_output=$(git merge-tree --write-tree "$target_commit" "$source_commit" 2>&1); then
    printf '%s\n' "$merge_output" >&2
    fail '目标分支与原项目合并失败或发生冲突；两个远程均未推送，请先处理冲突。'
  fi
  merge_tree=${merge_output%%$'\n'*}
  [[ $(git cat-file -t "$merge_tree" 2>/dev/null) == tree ]] || fail '无法获取合并结果；请检查 Git 版本（需要 2.38 或更新）。'
  result_commit=$(printf '%s\n' 'Merge TaskB main into ARM_to-placement_YE' | \
    git commit-tree "$merge_tree" -p "$target_commit" -p "$source_commit")
fi

# 跨仓库推送不是原子操作。先推原仓库，若第二步失败可安全重试。
git push origin "$source_commit:$source_ref" || fail '原仓库推送失败，目标仓库未推送。'
git push placement "$result_commit:$target_ref" || \
  fail '原仓库已推送，但目标分支推送失败；检查远端变化或权限后重试。'
printf '同步完成：原仓库 main %s；目标分支 ARM_to-placement_YE %s。\n' "$source_commit" "$result_commit"
