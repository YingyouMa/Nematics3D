from pathlib import Path


def test_repair_repository_tools_patch_stdin() -> None:
    source = Path(__file__).parents[1] / "src" / "nematics3d_mcp" / "repository_tools.py"
    text = source.read_text(encoding="utf-8")
    old = '''    common_arguments = {
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
'''
    new = '''    if input_text is None:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        input=input_text.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return subprocess.CompletedProcess(
        result.args,
        result.returncode,
        result.stdout.decode("utf-8", errors="replace"),
        result.stderr.decode("utf-8", errors="replace"),
    )
'''
    if old not in text:
        assert new in text
        return
    source.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")
    assert new in source.read_text(encoding="utf-8")
