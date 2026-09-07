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
- `run_hpcc_command`
- `list_caspar_files`
- `read_caspar_file`
- `backup_local_to_caspar`
- `backup_hpcc_to_caspar`

Caspar access is fixed to the working root reached by
`cd ../../current2/yingyou`. Inspection paths cannot escape that root. Backups
are placed in unique timestamped directories below `backups/`. Local backup
sources are restricted to one selected file or subdirectory in this Nematics3D
checkout; HPCC backup sources are restricted to one selected path below
`/work/yingyouma`. HPCC-to-Caspar copies use this computer as an `scp -3` relay.

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
open while ChatGPT uses the MCP server. Press Ctrl+C to stop the current tunnel.
Whether the tunnel stops manually or unexpectedly, the launcher retries after
30 seconds. During that delay, press `R` to retry immediately or `C` to stop the
reconnect loop. While the process is running, the launcher checks the tunnel's
control-plane poll health after a 45-second startup grace period and then every
10 seconds. Three consecutive unhealthy checks, including a last successful poll
older than 90 seconds, force a restart. Closing the terminal window stops the
launcher and tunnel.
`commit_changes` commits only the explicitly selected paths and never pushes.


## Controlled push

`push_current_branch(expected_commit)` pushes the verified current HEAD to the
same branch on `origin`. It requires the full expected commit SHA, never
force-pushes, and does not accept arbitrary remotes, branches, refspecs, or Git
arguments.

