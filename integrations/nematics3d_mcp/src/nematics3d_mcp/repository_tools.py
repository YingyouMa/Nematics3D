"""Repository-scoped operations exposed by the Nematics3D MCP server."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal


REPO_ROOT = Path(r"D:\Document\GitHub\Nematics3D").resolve()
CONDA_EXECUTABLE = Path(r"C:\Users\myy23\anaconda3\Scripts\conda.exe")
CONDA_ENVIRONMENT = "Nematics3D"

MAX_FILE_BYTES = 1_000_000
MAX_LIST_RESULTS = 1_000
MAX_SEARCH_RESULTS = 200
MAX_COMMAND_OUTPUT_CHARS = 60_000

IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".black-cache",
    "build",
    "dist",
    "dist_test",
}

ProjectTask = Literal["pytest", "black", "ruff", "build"]


def resolve_repo_path(relative_path: str = "") -> Path:
    """Resolve a user-provided path and keep it inside the repository."""
    if "\x00" in relative_path:
        raise ValueError("Paths cannot contain null bytes.")

    candidate = Path(relative_path)
    if candidate.is_absolute() or PureWindowsPath(relative_path).is_absolute():
        raise ValueError("Use a path relative to the Nematics3D repository.")

    resolved = (REPO_ROOT / candidate).resolve(strict=False)
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError("Path escapes the Nematics3D repository.") from error

    if ".git" in relative.parts:
        raise ValueError("Access to .git is forbidden.")

    return resolved


def list_files(path: str = "", pattern: str = "*") -> dict[str, object]:
    """List repository files below a relative path."""
    base = resolve_repo_path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates = [base] if base.is_file() else base.rglob(pattern)
    files: list[str] = []
    is_truncated = False

    for candidate in candidates:
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(REPO_ROOT)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        files.append(relative.as_posix())
        if len(files) >= MAX_LIST_RESULTS:
            is_truncated = True
            break

    files.sort()
    return {
        "path": Path(path).as_posix(),
        "pattern": pattern,
        "files": files,
        "count": len(files),
        "is_truncated": is_truncated,
    }


def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, object]:
    """Read a UTF-8 repository file, optionally restricting the line range."""
    file_path = resolve_repo_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    if file_path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the {MAX_FILE_BYTES}-byte read limit.")

    if start_line is not None and start_line < 1:
        raise ValueError("start_line must be at least 1.")
    if end_line is not None and end_line < 1:
        raise ValueError("end_line must be at least 1.")
    if start_line is not None and end_line is not None and end_line < start_line:
        raise ValueError("end_line cannot be smaller than start_line.")

    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Only UTF-8 text files can be read.") from error

    lines = text.splitlines(keepends=True)
    first = 1 if start_line is None else start_line
    last = len(lines) if end_line is None else min(end_line, len(lines))
    content = "".join(lines[first - 1 : last])

    return {
        "path": file_path.relative_to(REPO_ROOT).as_posix(),
        "content": content,
        "start_line": first,
        "end_line": last,
        "total_lines": len(lines),
    }


def search_text(
    query: str,
    path: str = "",
    pattern: str = "*.py",
) -> dict[str, object]:
    """Search UTF-8 repository files for a case-insensitive literal string."""
    if not query:
        raise ValueError("query cannot be empty.")

    base = resolve_repo_path(path)
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates = [base] if base.is_file() else base.rglob(pattern)
    query_folded = query.casefold()
    matches: list[dict[str, object]] = []
    is_truncated = False

    for candidate in candidates:
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(REPO_ROOT)
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if candidate.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            lines = candidate.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(lines, start=1):
            if query_folded not in line.casefold():
                continue
            matches.append(
                {
                    "path": relative.as_posix(),
                    "line": line_number,
                    "text": line,
                }
            )
            if len(matches) >= MAX_SEARCH_RESULTS:
                is_truncated = True
                break
        if is_truncated:
            break

    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "is_truncated": is_truncated,
    }


def _normalize_patch_path(raw_path: str) -> str | None:
    """Normalize one path extracted from a unified diff header."""
    if raw_path == "/dev/null":
        return None

    path = raw_path
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]

    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    if posix_path.is_absolute() or windows_path.is_absolute():
        raise ValueError(f"Patch path must be repository-relative: {raw_path}")
    if ".." in posix_path.parts or ".git" in posix_path.parts:
        raise ValueError(f"Unsafe patch path: {raw_path}")

    resolve_repo_path(path)
    return path


def extract_patch_paths(patch: str) -> list[str]:
    """Extract and validate paths from a Git-style unified diff."""
    if not patch.strip():
        raise ValueError("patch cannot be empty.")
    if "GIT binary patch" in patch or "Binary files " in patch:
        raise ValueError("Binary patches are not supported.")

    paths: set[str] = set()
    for line in patch.splitlines():
        raw_paths: list[str] = []
        if line.startswith("diff --git "):
            fields = shlex.split(line)
            if len(fields) != 4:
                raise ValueError("Malformed diff --git header.")
            raw_paths.extend(fields[2:4])
        elif line.startswith("--- ") or line.startswith("+++ "):
            fields = shlex.split(line)
            if len(fields) < 2:
                raise ValueError("Malformed unified diff file header.")
            raw_paths.append(fields[1])

        for raw_path in raw_paths:
            normalized = _normalize_patch_path(raw_path)
            if normalized is not None:
                paths.add(normalized)

    if not paths:
        raise ValueError("No repository file paths were found in the patch.")
    return sorted(paths)


def _run_process(
    command: list[str],
    *,
    input_text: str | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a command without a shell from the repository root."""
    common_arguments = {
        "args": command,
        "cwd": REPO_ROOT,
        "text": True,
        "capture_output": True,
        "timeout": timeout,
        "check": False,
    }
    if input_text is None:
        return subprocess.run(stdin=subprocess.DEVNULL, **common_arguments)
    return subprocess.run(input=input_text, **common_arguments)


def _bounded_output(text: str) -> tuple[str, bool]:
    """Limit command output returned to the model."""
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text, False
    return text[:MAX_COMMAND_OUTPUT_CHARS], True


def apply_patch(patch: str) -> dict[str, object]:
    """Validate and apply a Git-style unified diff to the repository."""
    paths = extract_patch_paths(patch)
    check_result = _run_process(
        ["git", "apply", "--check", "--whitespace=nowarn", "-"],
        input_text=patch,
    )
    if check_result.returncode != 0:
        raise ValueError(check_result.stderr.strip() or "git apply --check failed.")

    apply_result = _run_process(
        ["git", "apply", "--whitespace=nowarn", "-"],
        input_text=patch,
    )
    if apply_result.returncode != 0:
        raise RuntimeError(apply_result.stderr.strip() or "git apply failed.")

    status_result = _run_process(["git", "status", "--short", "--", *paths])
    diff_result = _run_process(["git", "diff", "--", *paths])
    diff, is_diff_truncated = _bounded_output(diff_result.stdout)

    return {
        "applied": True,
        "paths": paths,
        "status": status_result.stdout,
        "diff": diff,
        "is_diff_truncated": is_diff_truncated,
    }


def get_git_changes(path: str | None = None) -> dict[str, object]:
    """Return Git status and unstaged diff for the repository or one path."""
    paths: list[str] = []
    if path is not None:
        resolved = resolve_repo_path(path)
        paths.append(resolved.relative_to(REPO_ROOT).as_posix())

    separator = ["--", *paths] if paths else []
    status_result = _run_process(["git", "status", "--short", *separator])
    diff_result = _run_process(["git", "diff", *separator])
    diff, is_diff_truncated = _bounded_output(diff_result.stdout)

    return {
        "status": status_result.stdout,
        "diff": diff,
        "is_diff_truncated": is_diff_truncated,
    }


def commit_changes(message: str, paths: list[str]) -> dict[str, object]:
    """Commit only explicitly selected repository paths without pushing."""
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Commit message cannot be empty.")
    if "\n" in clean_message or "\r" in clean_message:
        raise ValueError("Commit message must be a single line.")
    if len(clean_message) > 200:
        raise ValueError("Commit message cannot exceed 200 characters.")
    if not paths:
        raise ValueError("At least one explicit repository path is required.")

    validated_paths = _validate_task_paths(paths)
    if any("::" in path for path in validated_paths):
        raise ValueError("Git commit paths cannot contain pytest node selectors.")

    selected_status = _run_process(["git", "status", "--short", "--", *validated_paths])
    if selected_status.returncode != 0:
        raise RuntimeError(selected_status.stderr.strip() or "git status failed.")
    if not selected_status.stdout.strip():
        raise ValueError("The selected paths contain no changes to commit.")

    stage_result = _run_process(["git", "add", "--", *validated_paths])
    if stage_result.returncode != 0:
        raise RuntimeError(stage_result.stderr.strip() or "git add failed.")

    commit_result = _run_process(
        [
            "git",
            "commit",
            "--only",
            "-m",
            clean_message,
            "--",
            *validated_paths,
        ]
    )
    if commit_result.returncode != 0:
        raise RuntimeError(commit_result.stderr.strip() or "git commit failed.")

    revision_result = _run_process(["git", "rev-parse", "HEAD"])
    summary_result = _run_process(
        ["git", "show", "--stat", "--oneline", "--no-renames", "HEAD"]
    )
    remaining_status = _run_process(["git", "status", "--short"])

    return {
        "committed": True,
        "commit": revision_result.stdout.strip(),
        "message": clean_message,
        "paths": validated_paths,
        "summary": summary_result.stdout,
        "remaining_status": remaining_status.stdout,
        "pushed": False,
    }


def _validate_task_paths(paths: list[str] | None) -> list[str]:
    """Validate task path arguments while preserving pytest node selectors."""
    validated: list[str] = []
    for value in paths or []:
        file_part, separator, selector = value.partition("::")
        resolved = resolve_repo_path(file_part)
        relative = resolved.relative_to(REPO_ROOT).as_posix()
        validated.append(relative + (separator + selector if separator else ""))
    return validated


def run_project_task(
    task: ProjectTask,
    paths: list[str] | None = None,
) -> dict[str, object]:
    """Run one allowlisted Nematics3D development task."""
    validated_paths = _validate_task_paths(paths)
    if not CONDA_EXECUTABLE.is_file():
        raise FileNotFoundError(f"Conda executable not found: {CONDA_EXECUTABLE}")

    task_arguments: dict[ProjectTask, list[str]] = {
        "pytest": ["python", "-m", "pytest"],
        "black": ["python", "-m", "black"],
        "ruff": ["python", "-m", "ruff", "check"],
        "build": ["python", "-m", "build"],
    }
    if task not in task_arguments:
        raise ValueError(f"Unsupported project task: {task}")
    if task == "build" and validated_paths:
        raise ValueError("The build task does not accept paths.")

    command = [
        str(CONDA_EXECUTABLE),
        "run",
        "-n",
        CONDA_ENVIRONMENT,
        *task_arguments[task],
        *validated_paths,
    ]
    result = _run_process(command)
    stdout, is_stdout_truncated = _bounded_output(result.stdout)
    stderr, is_stderr_truncated = _bounded_output(result.stderr)

    return {
        "task": task,
        "paths": validated_paths,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "is_output_truncated": is_stdout_truncated or is_stderr_truncated,
    }
