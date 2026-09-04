# Nematics3D MCP

Private MCP connection layer for the local Nematics3D repository.

The initial server intentionally exposes no project tools. Its first milestone is
to validate the stdio MCP handshake through OpenAI Secure MCP Tunnel.

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
open while ChatGPT uses the MCP server; press Ctrl+C to stop it.
