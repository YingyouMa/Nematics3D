"""stdio entry point for the private Nematics3D MCP server."""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from nematics3d_mcp import repository_tools


SERVER_NAME = "nematics3d-local"

mcp = FastMCP(
    SERVER_NAME,
    instructions=(
        "This private server is dedicated to the local Nematics3D repository. "
        "Use repository-relative paths. Inspect relevant files before editing, "
        "apply focused patches, preserve unrelated user changes, and report the "
        "resulting Git diff and validation results."
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


@mcp.tool(
    title="List Nematics3D repository files",
    description="List files below a repository-relative path using a glob pattern.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def list_files(path: str = "", pattern: str = "*") -> dict[str, object]:
    """List repository files."""
    return repository_tools.list_files(path, pattern)


@mcp.tool(
    title="Read a Nematics3D repository file",
    description=(
        "Read a UTF-8 text file from the repository, optionally selecting a "
        "one-based inclusive line range."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict[str, object]:
    """Read a repository file."""
    return repository_tools.read_file(path, start_line, end_line)


@mcp.tool(
    title="Search Nematics3D repository text",
    description=(
        "Search UTF-8 repository files for a case-insensitive literal string."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def search_text(
    query: str,
    path: str = "",
    pattern: str = "*.py",
) -> dict[str, object]:
    """Search repository text."""
    return repository_tools.search_text(query, path, pattern)


@mcp.tool(
    title="Apply a patch to Nematics3D",
    description=(
        "Create, modify, or delete repository files by applying a Git-style "
        "unified diff. Paths outside the repository and .git are forbidden."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
def apply_patch(patch: str) -> dict[str, object]:
    """Apply a validated unified diff."""
    return repository_tools.apply_patch(patch)


@mcp.tool(
    title="Get Nematics3D Git changes",
    description="Return Git status and unstaged diff for the repository or one path.",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
def get_git_changes(path: str | None = None) -> dict[str, object]:
    """Inspect current Git changes."""
    return repository_tools.get_git_changes(path)


@mcp.tool(
    title="Commit selected Nematics3D changes",
    description=(
        "Create a local Git commit containing only explicitly selected repository "
        "paths. This tool never stages the whole repository and never pushes."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
def commit_changes(message: str, paths: list[str]) -> dict[str, object]:
    """Commit explicitly selected repository changes."""
    return repository_tools.commit_changes(message, paths)


@mcp.tool(
    title="Run an approved Nematics3D project task",
    description=(
        "Run pytest, Black, Ruff, or package build in the Nematics3D conda "
        "environment. Arbitrary shell commands are not accepted."
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
def run_project_task(
    task: repository_tools.ProjectTask,
    paths: list[str] | None = None,
) -> dict[str, object]:
    """Run one allowlisted project task."""
    return repository_tools.run_project_task(task, paths)


def main() -> None:
    """Run the MCP server over standard input and standard output."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
