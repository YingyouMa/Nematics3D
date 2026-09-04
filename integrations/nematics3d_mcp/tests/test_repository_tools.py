"""Focused tests for repository-scoped MCP operations."""

from pathlib import Path

import pytest

from nematics3d_mcp import repository_tools


@pytest.fixture
def temporary_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point repository tools at a temporary Git worktree."""
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(repository_tools, "REPO_ROOT", repository.resolve())
    repository_tools._run_process(["git", "init"])
    return repository


def test_resolve_repo_path_rejects_escape(temporary_repository: Path) -> None:
    """Parent traversal cannot escape the configured repository root."""
    with pytest.raises(ValueError, match="escapes"):
        repository_tools.resolve_repo_path("../outside.txt")


def test_resolve_repo_path_rejects_git_directory(
    temporary_repository: Path,
) -> None:
    """Git internals remain inaccessible."""
    with pytest.raises(ValueError, match=".git"):
        repository_tools.resolve_repo_path(".git/config")


def test_apply_patch_modifies_repository_file(temporary_repository: Path) -> None:
    """A valid unified diff changes a file inside the repository."""
    target = temporary_repository / "example.py"
    target.write_text("value = 1\n", encoding="utf-8")

    result = repository_tools.apply_patch(
        """diff --git a/example.py b/example.py
--- a/example.py
+++ b/example.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
    )

    assert result["applied"] is True
    assert result["paths"] == ["example.py"]
    assert target.read_text(encoding="utf-8") == "value = 2\n"


def test_apply_patch_accepts_begin_patch_update(temporary_repository: Path) -> None:
    """The model's Begin Patch update format changes an existing file."""
    target = temporary_repository / "example.py"
    target.write_text("value = 1\nother = 3\n", encoding="utf-8")

    result = repository_tools.apply_patch(
        """*** Begin Patch
*** Update File: example.py
@@
-value = 1
+value = 2
 other = 3
*** End Patch
"""
    )

    assert result["paths"] == ["example.py"]
    assert target.read_text(encoding="utf-8") == "value = 2\nother = 3\n"


def test_apply_patch_accepts_begin_patch_add_and_delete(
    temporary_repository: Path,
) -> None:
    """The wrapped format can create one file and delete another."""
    removed = temporary_repository / "removed.txt"
    removed.write_text("remove me\n", encoding="utf-8")

    result = repository_tools.apply_patch(
        """*** Begin Patch
*** Add File: added.txt
+created
*** Delete File: removed.txt
*** End Patch
"""
    )

    assert result["paths"] == ["added.txt", "removed.txt"]
    assert (temporary_repository / "added.txt").read_text(
        encoding="utf-8"
    ) == "created\n"
    assert not removed.exists()


def test_begin_patch_rejects_unsafe_path(temporary_repository: Path) -> None:
    """Wrapped patches retain the repository path boundary."""
    with pytest.raises(ValueError, match="Unsafe patch path"):
        repository_tools.apply_patch(
            """*** Begin Patch
*** Add File: ../outside.txt
+unsafe
*** End Patch
"""
        )


def test_extract_patch_paths_rejects_git_directory() -> None:
    """A patch cannot target Git internals."""
    with pytest.raises(ValueError, match="Unsafe patch path"):
        repository_tools.extract_patch_paths(
            """diff --git a/.git/config b/.git/config
--- a/.git/config
+++ b/.git/config
@@ -1 +1 @@
-old
+new
"""
        )


def test_commit_changes_commits_only_selected_paths(
    temporary_repository: Path,
) -> None:
    """A controlled commit leaves unrelated worktree changes untouched."""
    repository_tools._run_process(["git", "config", "user.name", "MCP Test"])
    repository_tools._run_process(
        ["git", "config", "user.email", "mcp-test@example.invalid"]
    )
    selected = temporary_repository / "selected.txt"
    unrelated = temporary_repository / "unrelated.txt"
    selected.write_text("selected\n", encoding="utf-8")
    unrelated.write_text("unrelated\n", encoding="utf-8")

    result = repository_tools.commit_changes(
        "Test selected commit",
        ["selected.txt"],
    )

    assert result["committed"] is True
    assert result["paths"] == ["selected.txt"]
    assert result["pushed"] is False
    assert "?? unrelated.txt" in result["remaining_status"]


def test_push_current_branch_pushes_only_verified_head(
    temporary_repository: Path,
) -> None:
    """The controlled push updates only the matching branch on origin."""
    repository_tools._run_process(["git", "config", "user.name", "MCP Test"])
    repository_tools._run_process(
        ["git", "config", "user.email", "mcp-test@example.invalid"]
    )
    repository_tools._run_process(["git", "checkout", "-b", "develop"])

    target = temporary_repository / "example.txt"
    target.write_text("example\n", encoding="utf-8")
    repository_tools._run_process(["git", "add", "--", "example.txt"])
    repository_tools._run_process(["git", "commit", "-m", "Initial commit"])
    head = repository_tools._run_process(["git", "rev-parse", "HEAD"]).stdout.strip()

    remote = temporary_repository.parent / "remote.git"
    repository_tools._run_process(["git", "init", "--bare", str(remote)])
    repository_tools._run_process(["git", "remote", "add", "origin", str(remote)])

    result = repository_tools.push_current_branch(head)

    assert result["pushed"] is True
    assert result["remote"] == "origin"
    assert result["branch"] == "develop"
    assert result["commit"] == head
    remote_head = repository_tools._run_process(
        ["git", "--git-dir", str(remote), "rev-parse", "refs/heads/develop"]
    )
    assert remote_head.stdout.strip() == head


def test_push_current_branch_rejects_unexpected_head(
    temporary_repository: Path,
) -> None:
    """A stale or invented expected SHA cannot push repository state."""
    repository_tools._run_process(["git", "config", "user.name", "MCP Test"])
    repository_tools._run_process(
        ["git", "config", "user.email", "mcp-test@example.invalid"]
    )
    target = temporary_repository / "example.txt"
    target.write_text("example\n", encoding="utf-8")
    repository_tools._run_process(["git", "add", "--", "example.txt"])
    repository_tools._run_process(["git", "commit", "-m", "Initial commit"])

    with pytest.raises(ValueError, match="does not match"):
        repository_tools.push_current_branch("0" * 40)
