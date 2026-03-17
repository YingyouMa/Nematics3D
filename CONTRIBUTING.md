# Contributing to Nematics3D

This document is the developer-facing entry point for contributing to the Nematics3D repository.

## Development principles

- Preserve existing repository conventions when modifying code.
- Prefer local consistency with nearby code over introducing a new style.
- Use automated formatting for normal Python code.
- Keep intentionally structured metadata blocks readable as visual lookup tables.

## Formatting

Run `black` after making code changes.

Treat `black` as the default formatter for regular Python code, especially:

- function bodies
- control flow
- procedural logic
- algorithmic code

Do not manually fight `black` in these areas.

## Aligned metadata blocks

Some parts of this repository are intentionally written as visually aligned declarative tables instead of ordinary Black-style formatting.

This applies especially to short, high-signal blocks such as:

- dataclass field declarations
- `__attrs__`
- `__relations__`
- `_validators`
- `_DEFAULTS_FROZEN`
- similar metadata, schema, or option dictionaries

When this layout makes the code easier to scan, keep the alignment and protect the block with:

```py
# fmt: off
...
# fmt: on
```

Use this narrowly and only for compact declarative blocks whose readability depends on alignment.

## When not to use `fmt: off`

Do not use `# fmt: off` for:

- normal implementation code
- long functions
- large procedural sections
- control-heavy logic
- long unstable blocks that are difficult to maintain manually

If alignment does not clearly improve readability, let `black` format the code normally.

## Override comment blocks

When a method intentionally overrides inherited behavior, add an override comment block immediately above the method using the repository pattern:

```py
# ==================== OVERRIDE ====================
# ClassName overrides BaseClass.method_name because ...
# ==================================================
```

This is part of the repository code style for real behavioral overrides. Use it to make extension points and changed semantics obvious to later readers.
## Detailed style guide

For more detailed repository conventions, see [docs/code-style.md](/D:/Document/GitHub/Nematics3D/docs/code-style.md).
