# 本分支的 TaskB 来源

- 来源仓库：<https://github.com/yxh0527-ship-it/taskb> 的 `main` 分支，提交 `ba1f01f7db8e5e783113ba3834997f26a103670d`。
- 本分支以团队仓库 `main` 的提交 `e02a7035fe703d4f4bd05fadb72f2234ae15ae8e` 为基础，导入来源仓库已提交的文件；团队协作规则见 [README](README)。
- 来源仓库的 [docs/git-workflow.md](docs/git-workflow.md) 描述其原有服务器和 Git 工作流。在团队仓库操作时，请先用 `git remote -v` 和 `git branch --show-current` 确认目标，再遵循团队分支协作规则。
- 根目录的 `README_TaskB_with_backet.md` 和 `RELEASE_MANIFEST.json` 来自团队仓库原有的 2026-10-23 发布版本。该清单只对应当时的发布包，不是本分支当前开发目录的文件校验清单。

正式 Isaac Sim/PhysX 运行仍需在兼容的环境中单独验证。离线检查与已提供的模型文件不能证明完整任务闭环已经完成。
