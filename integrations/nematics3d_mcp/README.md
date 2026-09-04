# Nematics3D MCP

Private MCP connection layer for the local Nematics3D repository. It exposes
repository-scoped file browsing, text search, unified-diff patching, Git change
inspection, and a small allowlist of project validation tasks.

All paths are repository-relative. Access to `.git` and paths outside the
Nematics3D checkout is rejected. Project commands run without a shell.

## MCP tools

- `get_server_status`
- `list_files`
- `read_file`
- `search_text`
- `apply_patch`
- `get_git_changes`
- `commit_changes`
- `run_project_task`

## Local development

Use Python 3.12 and install this package into an isolated environment:

```powershell
python -m pip install -e .
nematics3d-mcp
```

The process communicates through standard input and standard output. Do not write
diagnostic output to standard output because that channel is reserved for MCP
JSON-RPC messages.

## Tunnel

The OpenAI tunnel identifier is configured outside the repository. API keys and
generated tunnel-client profiles must never be committed.

## One-click Windows startup

Run the one-time setup from a PowerShell session that already has
`CONTROL_PLANE_API_KEY` set:

```powershell
.\scripts\setup-one-click.ps1
```

If that environment variable is absent, the setup securely prompts for the key.
It installs the tunnel-client runtime under the current user's local application
data and protects the key with Windows DPAPI. The encrypted value can only be
decrypted by the same Windows user on the same computer.

After setup, double-click `start-nematics3d-mcp.cmd`. Keep its terminal window
open while ChatGPT uses the MCP server. Press Ctrl+C to stop the tunnel, then
enter `R` to restart it in the same window or `Q` to quit.
`commit_changes` commits only the explicitly selected paths and never pushes.


## Controlled push

`push_current_branch(expected_commit)` pushes the verified current HEAD to the
same branch on `origin`. It requires the full expected commit SHA, never
force-pushes, and does not accept arbitrary remotes, branches, refspecs, or Git
arguments.

