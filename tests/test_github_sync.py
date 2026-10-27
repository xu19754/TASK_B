"""Exercise the two-repository workflow against disposable local Git remotes."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_github.sh"
ORIGIN_URL = "https://github.com/yxh0527-ship-it/taskb.git"
TEAM_URL = "https://github.com/xu19754/TASK_B.git"
TARGET = "ARM_to-placement_YE"


class GitHubSyncTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = self.root / "origin.git"
        self.team = self.root / "team.git"
        self.work = self.root / "work"
        self.environment = os.environ.copy()
        self.environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0")
        for bare in (self.origin, self.team):
            self.run_git(self.root, "init", "--bare", "--initial-branch=main", str(bare))
        self.run_git(self.root, "init", "--initial-branch=main", str(self.work))
        self.identity(self.work)
        self.write_commit(self.work, "shared.txt", "base\n", "initial")
        self.run_git(self.work, "remote", "add", "origin", ORIGIN_URL)
        self.run_git(self.work, "remote", "add", "placement", TEAM_URL)
        self.run_git(self.work, "config", f"url.{self.origin.as_uri()}.insteadOf", ORIGIN_URL)
        self.run_git(self.work, "config", f"url.{self.team.as_uri()}.insteadOf", TEAM_URL)
        self.run_git(self.work, "push", "origin", "HEAD:refs/heads/main")
        for branch in (TARGET, "main", "other"):
            self.run_git(self.work, "push", "placement", f"HEAD:refs/heads/{branch}")

    def run_git(self, cwd, *args, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check,
        )

    def identity(self, cwd):
        self.run_git(cwd, "config", "user.name", "Sync Test")
        self.run_git(cwd, "config", "user.email", "sync@example.invalid")

    def write_commit(self, cwd, name, content, message):
        (cwd / name).write_text(content, encoding="utf-8")
        self.run_git(cwd, "add", name)
        self.run_git(cwd, "commit", "-m", message)

    def ref(self, bare, branch):
        return self.run_git(bare, "rev-parse", f"refs/heads/{branch}").stdout.strip()

    def run_sync(self):
        return subprocess.run(
            ["bash", str(SCRIPT)], cwd=self.work, env=self.environment, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def team_checkout(self):
        checkout = self.root / "team-work"
        self.run_git(self.root, "clone", "--branch", TARGET, str(self.team), str(checkout))
        self.identity(checkout)
        return checkout

    def test_merges_both_histories_and_does_not_change_other_branches(self):
        team_work = self.team_checkout()
        self.write_commit(team_work, "team.txt", "team only\n", "team work")
        self.run_git(team_work, "push", "origin", f"HEAD:refs/heads/{TARGET}")
        old_target = self.ref(self.team, TARGET)
        old_other_refs = {name: self.ref(self.team, name) for name in ("main", "other")}
        self.write_commit(self.work, "shared.txt", "source update\n", "source work")
        source_head = self.run_git(self.work, "rev-parse", "HEAD").stdout.strip()

        result = self.run_sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.ref(self.origin, "main"), source_head)
        self.assertEqual({name: self.ref(self.team, name) for name in old_other_refs}, old_other_refs)
        self.assertEqual(self.run_git(self.team, "show", f"{TARGET}:shared.txt").stdout, "source update\n")
        self.assertEqual(self.run_git(self.team, "show", f"{TARGET}:team.txt").stdout, "team only\n")
        self.assertEqual(self.run_git(self.team, "merge-base", "--is-ancestor", old_target, TARGET, check=False).returncode, 0)
        self.assertEqual(self.run_git(self.team, "merge-base", "--is-ancestor", source_head, TARGET, check=False).returncode, 0)
        merged_head = self.ref(self.team, TARGET)
        self.assertEqual(self.run_sync().returncode, 0)
        self.assertEqual(self.ref(self.team, TARGET), merged_head)

    def test_conflict_stops_before_either_push(self):
        team_work = self.team_checkout()
        self.write_commit(team_work, "shared.txt", "team change\n", "team change")
        self.run_git(team_work, "push", "origin", f"HEAD:refs/heads/{TARGET}")
        self.write_commit(self.work, "shared.txt", "source change\n", "source change")
        before = (self.ref(self.origin, "main"), self.ref(self.team, TARGET))
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONFLICT", result.stderr)
        self.assertEqual((self.ref(self.origin, "main"), self.ref(self.team, TARGET)), before)

    def test_dirty_worktree_and_wrong_origin_are_rejected(self):
        (self.work / "untracked.txt").write_text("not committed", encoding="utf-8")
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("未提交", result.stderr)
        (self.work / "untracked.txt").unlink()
        self.run_git(self.work, "remote", "set-url", "origin", TEAM_URL)
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不是预期仓库", result.stderr)

    def test_source_remote_changes_are_not_overwritten(self):
        other_work = self.root / "origin-work"
        self.run_git(self.root, "clone", str(self.origin), str(other_work))
        self.identity(other_work)
        self.write_commit(other_work, "remote.txt", "remote change\n", "remote change")
        self.run_git(other_work, "push", "origin", "HEAD:refs/heads/main")
        self.write_commit(self.work, "local.txt", "local change\n", "local change")
        before = (self.ref(self.origin, "main"), self.ref(self.team, TARGET))
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("原仓库 main 有本地未包含的提交", result.stderr)
        self.assertEqual((self.ref(self.origin, "main"), self.ref(self.team, TARGET)), before)

    def test_retries_after_only_the_first_push_succeeds(self):
        old_target = self.ref(self.team, TARGET)
        self.write_commit(self.work, "shared.txt", "new source\n", "source update")
        head = self.run_git(self.work, "rev-parse", "HEAD").stdout.strip()
        hook = self.team / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        hook.chmod(0o755)
        result = self.run_sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("原仓库已推送", result.stderr)
        self.assertEqual(self.ref(self.origin, "main"), head)
        self.assertEqual(self.ref(self.team, TARGET), old_target)
        hook.unlink()
        retry = self.run_sync()
        self.assertEqual(retry.returncode, 0, retry.stderr)
        self.assertEqual(self.ref(self.team, TARGET), head)


if __name__ == "__main__":
    unittest.main()
