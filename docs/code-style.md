# Nematics3D Code Style

This document records repository-specific coding conventions for Nematics3D.

It is intended for human contributors. Agent-specific guidance may also exist in repository automation files and Codex skills, but the rules here should remain readable and useful on their own.

## 1. Default formatting rule

Use `black` as the default formatter for Python code.

After editing code, run `black` on the files you changed.

For ordinary Python implementation code, accept Black's formatting rather than manually preserving a custom layout.

This includes:

- function bodies
- conditional logic
- loops
- exception handling
- algorithmic code
- general procedural code

## 2. Structured declarative blocks

Some blocks in this repository are more readable when treated as compact visual tables rather than plain auto-formatted Python.

Typical examples include:

- dataclass field declarations
- `__attrs__`
- `__relations__`
- `_validators`
- `_DEFAULTS_FROZEN`
- similar metadata, schema, registry, or option-definition mappings

When these blocks are short and stable, prefer an aligned layout that helps readers compare entries vertically.

The intent is to make the block easier to scan as a reference structure:

- keys line up vertically
- types, defaults, or descriptions line up visually
- related items remain grouped
- the block reads like a schema or lookup table

## 3. Protect aligned blocks with `fmt` markers

When a declarative block is intentionally aligned, preserve that layout with a local formatting guard:

```py
# fmt: off
...
# fmt: on
```

Use the guard only around the specific block that needs protection.

Do not disable formatting for unrelated surrounding code.

## 4. Decision rule for aligned formatting

Use aligned table-style formatting only when all of the following are true:

- the code is primarily declarative rather than procedural
- the block is reasonably short
- alignment materially improves human scanability
- preserving the layout is worth local `fmt` protection

Otherwise, use normal Black formatting.

## 5. When not to use `# fmt: off`

Do not use `# fmt: off` for:

- normal function bodies
- long procedural code
- control-flow-heavy logic
- large regions where alignment adds little value
- blocks that change frequently and are costly to maintain manually

If the block is long, unstable, or not substantially clearer when aligned, let `black` format it normally.

## 6. Editing existing aligned blocks

When modifying an existing aligned block:

- preserve the established alignment style
- keep related entries grouped under existing headings or comments
- extend the local `fmt` protection if needed
- run `black` after the edit so the rest of the file stays normalized

## 7. Example pattern

The following is representative of the intended style for short declarative blocks:

```py
# fmt: off
_actor_attr: ClassVar[Mapping[str, str]] = {
    "is_visible":       "visibility",
    "is_pickable":      "pickable",
    "shading_type":     "prop.interpolation",
    "ambient":          "prop.ambient",
}
# fmt: on
```

This style is appropriate because the block is short, declarative, and easier to read when the mapping is visually aligned.
