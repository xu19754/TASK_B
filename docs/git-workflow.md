# Git 更新说明

原 GitHub 私有仓库：<https://github.com/yxh0527-ship-it/taskb> 的 `main`。

团队 GitHub 私有仓库：<https://github.com/xu19754/TASK_B/tree/ARM_to-placement_YE> 的 `ARM_to-placement_YE`。团队仓库的 `main` 和其他分支不在同步目标内。

服务器项目目录：`/root/gpufree-data/5810_20261023/TaskB`。

## 日常更新

在项目目录中执行。若其他电脑也修改此仓库，开始工作前先确认 `git status` 显示工作区干净，再运行 `git pull --ff-only`。

```bash
cd /root/gpufree-data/5810_20261023/TaskB
git status
git diff
git add .
git diff --cached --stat
git commit -m "说明本次修改"
bash scripts/sync_github.sh
```

保存文件和提交都不会自动上传；每次完成提交后，使用上述同步命令推送两个目标。普通 `git push` 仍只推送其配置的单个远程，不会自动同步团队分支。提交前检查暂存文件，不要上传密码、访问令牌、私钥或训练日志。

同步命令要求当前位于原项目的 `main` 且工作区干净，并需要 Git 2.38 或更新版本。命令先获取两个远程的指定分支，保留团队分支已有的提交和独有文件，将原项目提交合并到该分支；如果原仓库有尚未包含的提交或发生合并冲突，会停止并提示处理，不强制推送。两个仓库的推送不是原子操作：若原仓库推送成功而团队分支推送失败，检查报错后重新执行即可。

新克隆的原仓库需要配置团队远程，并使用对两个私有仓库均有权限的 GitHub 账号：

```bash
git remote add -t ARM_to-placement_YE placement https://github.com/xu19754/TASK_B.git
```

脚本只接受上述两个指定仓库，并且只推送原仓库的 `main` 和团队仓库的 `ARM_to-placement_YE`；请不要在团队仓库克隆中运行它。

`.gitignore` 已排除常见 Python 虚拟环境、缓存、macOS 元数据、环境密钥文件和项目根目录下的运行输出。原有代码、机器人模型、网格、底层 policy 与报告保留在版本控制中。其他位置生成的数据需按实际路径补充忽略规则。

## 在其他电脑继续工作

```bash
git clone https://github.com/yxh0527-ship-it/taskb.git
cd taskb
```

私有仓库要求使用有访问权限的 GitHub 账号完成认证。Python、Isaac Sim、Isaac Lab 等运行环境需要在目标机器上另外安装。

## 查看版本

```bash
git log --oneline -10
git status
```

推送失败时先解决具体报错；不要使用强制推送覆盖远端记录。超过 100 MiB 的单个文件应先配置 Git LFS 或采用独立存储，再加入提交。
