# Git 更新说明

GitHub 私有仓库：https://github.com/yxh0527-ship-it/taskb

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
git push
```

保存文件不会自动上传；提交保存本地版本，推送上传到 GitHub。提交前检查暂存文件，不要上传密码、访问令牌、私钥或训练日志。

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
