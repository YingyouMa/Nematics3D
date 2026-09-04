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
