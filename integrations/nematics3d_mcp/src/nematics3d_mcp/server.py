"""stdio entry point for the private Nematics3D MCP server."""

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "nematics3d-local"

mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "This private server is dedicated to the local Nematics3D repository. "
        "Its capabilities are intentionally empty while the connection is being set up."
    ),
)


def main() -> None:
    """Run the MCP server over standard input and standard output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
