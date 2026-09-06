"""Tests for the Brandeis HPCC SSH tool."""

import subprocess

import pytest

from nematics3d_mcp import hpcc_tools


def test_resolve_hpcc_working_directory_accepts_relative_path() -> None:
    """Relative paths resolve below the fixed HPCC work root."""
    result = hpcc_tools.resolve_hpcc_working_directory("projects/example")

    assert result.as_posix() == "/work/yingyouma/projects/example"


@pytest.mark.parametrize(
    "working_directory",
    ["/tmp", "../outside", "/work/yingyouma/../../etc"],
)
def test_resolve_hpcc_working_directory_rejects_escape(
    working_directory: str,
) -> None:
    """The caller cannot select a working directory outside the work root."""
    with pytest.raises(ValueError, match="must stay inside"):
        hpcc_tools.resolve_hpcc_working_directory(working_directory)


def test_run_hpcc_command_uses_noninteractive_ssh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The arbitrary command crosses SSH without invoking a local shell."""
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, "job submitted\n", "")

    monkeypatch.setattr(hpcc_tools.shutil, "which", lambda name: "ssh.exe")
    monkeypatch.setattr(hpcc_tools.subprocess, "run", fake_run)

    result = hpcc_tools.run_hpcc_command(
        "sbatch case.sh",
        "cases/test case",
        120,
    )

    assert captured["command"] == [
        "ssh.exe",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "yingyouma@hpcc.brandeis.edu",
        "cd -- '/work/yingyouma/cases/test case' && sbatch case.sh",
    ]
    assert captured["kwargs"] == {
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 120,
        "check": False,
    }
    assert result["exit_code"] == 0
    assert result["stdout"] == "job submitted\n"


def test_run_hpcc_command_rejects_invalid_timeout() -> None:
    """Remote commands cannot request an unbounded MCP call."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        hpcc_tools.run_hpcc_command("pwd", timeout_seconds=0)


def test_run_hpcc_command_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSH timeouts become concise tool errors."""
    monkeypatch.setattr(hpcc_tools.shutil, "which", lambda name: "ssh.exe")

    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("ssh", 2)

    monkeypatch.setattr(hpcc_tools.subprocess, "run", raise_timeout)

    with pytest.raises(TimeoutError, match="2-second timeout"):
        hpcc_tools.run_hpcc_command("sleep 10", timeout_seconds=2)
