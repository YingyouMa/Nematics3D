"""Controlled SSH entry points for Yingyou Ma's Brandeis HPCC account."""

from __future__ import annotations

import posixpath
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath


HPCC_DESTINATION = "yingyouma@hpcc.brandeis.edu"
HPCC_WORK_ROOT = PurePosixPath("/work/yingyouma")

MAX_COMMAND_CHARS = 20_000
MAX_COMMAND_OUTPUT_CHARS = 60_000
MAX_TIMEOUT_SECONDS = 3_600
SSH_CONNECT_TIMEOUT_SECONDS = 15


def resolve_hpcc_working_directory(
    working_directory: str = str(HPCC_WORK_ROOT),
) -> PurePosixPath:
    """Resolve a working directory and keep it below the HPCC work root."""
    if not working_directory:
        raise ValueError("working_directory cannot be empty.")
    if (
        "\x00" in working_directory
        or "\n" in working_directory
        or "\r" in working_directory
    ):
        raise ValueError("working_directory contains forbidden characters.")

    candidate = PurePosixPath(working_directory)
    if not candidate.is_absolute():
        candidate = HPCC_WORK_ROOT / candidate
    normalized = PurePosixPath(posixpath.normpath(candidate.as_posix()))

    try:
        normalized.relative_to(HPCC_WORK_ROOT)
    except ValueError as error:
        raise ValueError(
            f"working_directory must stay inside {HPCC_WORK_ROOT}."
        ) from error
    return normalized


def _bounded_output(text: str) -> tuple[str, bool]:
    """Limit SSH output returned through the MCP connection."""
    if len(text) <= MAX_COMMAND_OUTPUT_CHARS:
        return text, False
    return text[:MAX_COMMAND_OUTPUT_CHARS], True


def run_hpcc_command(
    command: str,
    working_directory: str = str(HPCC_WORK_ROOT),
    timeout_seconds: int = 300,
) -> dict[str, object]:
    """Run an arbitrary shell command through non-interactive SSH."""
    if not command.strip():
        raise ValueError("command cannot be empty.")
    if len(command) > MAX_COMMAND_CHARS:
        raise ValueError(f"command cannot exceed {MAX_COMMAND_CHARS} characters.")
    if not 1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise ValueError(
            f"timeout_seconds must be between 1 and {MAX_TIMEOUT_SECONDS}."
        )

    remote_directory = resolve_hpcc_working_directory(working_directory)
    ssh_executable = shutil.which("ssh")
    if ssh_executable is None:
        raise FileNotFoundError("The OpenSSH ssh executable was not found.")

    remote_command = f"cd -- {shlex.quote(remote_directory.as_posix())} && {command}"
    ssh_command = [
        ssh_executable,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        HPCC_DESTINATION,
        remote_command,
    ]

    try:
        result = subprocess.run(
            ssh_command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"HPCC command exceeded the {timeout_seconds}-second timeout."
        ) from error

    stdout, is_stdout_truncated = _bounded_output(result.stdout)
    stderr, is_stderr_truncated = _bounded_output(result.stderr)
    return {
        "destination": HPCC_DESTINATION,
        "working_directory": remote_directory.as_posix(),
        "command": command,
        "exit_code": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "is_output_truncated": is_stdout_truncated or is_stderr_truncated,
    }
