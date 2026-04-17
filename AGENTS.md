# Nematics3D Agent Notes

## Available skills

- `nematics3d-repo-conventions`
  - Location: `C:\Users\myy23\.codex\skills\nematics3d-repo-conventions\SKILL.md`
  - Use when working in this repository and repository-specific coding conventions should be followed.

## Default guidance

When modifying code in this repository, prefer using the `nematics3d-repo-conventions` skill before making changes.

## Startup required skills

At the beginning of work in this repository, read:

`C:\Users\myy23\.codex\skills\local-write-escalation\SKILL.md`

Use it whenever local file reads/writes, Windows permissions, sandbox failures,
`apply_patch` failures, Access Denied errors, or environment-specific file
editing are involved.

## Windows validation notes

When checking Python syntax on Windows, remember that `python -m py_compile`
writes `.pyc` files into `__pycache__`. If it fails with `Access Denied` or
`Permission denied` while writing `__pycache__`, classify that as a cache/write
permission problem first, not as a code syntax failure.

For syntax-only validation in that situation, prefer an in-memory compile check
that does not write `.pyc` files, for example:

```powershell
python -c "from pathlib import Path; compile(Path('path/to/file.py').read_text(encoding='utf-8'), 'path/to/file.py', 'exec')"
```

Use the project conda environment for the command when needed. Only clean or
delete `__pycache__` after confirming it is safe and relevant to the task.
