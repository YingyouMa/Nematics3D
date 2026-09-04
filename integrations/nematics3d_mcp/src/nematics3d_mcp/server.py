"""stdio entry point for the private Nematics3D MCP server."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


SERVER_NAME = "nematics3d-local"

mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "This private server is dedicated to the local Nematics3D repository. "
        "Its capabilities are intentionally empty while the connection is being set up."
    ),
)


@mcp.tool(
    title="Get Nematics3D MCP server status",
    description=(
        "Check whether the private local Nematics3D MCP server is reachable. "
        "This diagnostic does not read or modify repository files."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def get_server_status() -> dict[str, str]:
    """Return a minimal status payload for end-to-end connection testing."""
    return {
        "status": "ready",
        "server": SERVER_NAME,
        "transport": "stdio",
    }


def main() -> None:
    """Run the MCP server over standard input and standard output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
