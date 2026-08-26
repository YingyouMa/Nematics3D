# Tutorial Writing Guide

This guide records the writing and formatting conventions for `Nematics3D`
tutorials. Apply these rules to new tutorials and to existing tutorials when
they are substantially revised.

## Practical completion standard

This guide is a collection of writing tools, not a checklist that every
tutorial must satisfy item by item. Applying every possible section and rule
to every function can create disproportionate work and make a tutorial longer
than its subject warrants. Select the sections that materially help readers,
and omit optional sections that have no useful content.

When judgment is needed, use the
[`q_diagonalize()` tutorial](../analysis/q_diagonalization/q_diagonalize.ipynb) as the repository's
practical reference for expected completeness, structure, and restraint. A
tutorial does not need to exceed that example merely to demonstrate compliance
with this guide.

## Mathematical quantities in prose

Use LaTeX for mathematical and physical quantities whenever they appear in
tutorial prose. In particular, the Q-tensor symbol must always be rendered as
`$Q$`, just as the scalar order parameter and director are rendered as `$S$`
and `$\mathbf{n}$`.

Examples:

- Write "when $Q$ is the available representation", not "when Q is the
  available representation".
- Write "$Q$-tensor data", not "Q-tensor data".
- Write "convert $Q$ into $S$ and $\mathbf{n}$", not "convert Q into S and n".

This rule applies to headings, paragraphs, lists, table prose, captions, and
other explanatory text. It does not apply to code identifiers or literal code:

- keep `q_diagonalize()` as code;
- keep `QFieldObject` as code;
- keep a Python variable such as `Q_data` in a code span;
- do not put LaTeX inside executable code cells.

When a sentence refers to both the mathematical quantity and a Python object,
distinguish them explicitly. For example:

> The mathematical tensor $Q$ is stored in the Python array `Q_data`.

## Software and package names in prose

Whenever a repository, project, software package, or numerical library name
appears in tutorial prose, render it as inline code. Examples include
`Nematics3D`, `NumPy`, `NumExpr`, `SciPy`, `PyVista`, and `BLAS`. This makes
software names visually distinct from surrounding prose and keeps their
spelling consistent across tutorials.

This rule applies to headings, paragraphs, lists, tables, captions, and link
descriptions. Do not leave software names unformatted or use bold text or
italics in place of inline code.

Keep import names and longer API identifiers in inline code as well, but
preserve their exact spelling and capitalization:

- write "`Nematics3D` provides this convention", not "Nematics3D provides
  this convention";
- write "the `NumPy` implementation", not "the NumPy implementation";
- write "the `NumExpr` backend", not "the NumExpr backend";
- write "import the `nematics3d` package" when referring specifically to the
  Python package;
- keep identifiers such as `nematics3d.q_diagonalize()` in code style.

## Skippable setup sections

When a `Setup` section contains only mechanical preparation and readers can run
the tutorial without understanding those details, begin the section with this
bold notice:

> **For readers who are only interested in the tutorial, this section can be
> safely skipped.**

Continue on the same paragraph with a concise description of what the setup
cells do, such as importing dependencies, locating data, or creating an output
directory. The notice must be bold so a reader can recognize it immediately.

Do not use this notice when the section contains a prerequisite, convention, or
decision that the reader must understand to interpret the later examples.

## Developer-only sections

When a section is intended for developers rather than ordinary users, begin
the section with this bold notice immediately below its heading:

> **This section is intended for developers. Regular users can safely skip it.**

Use this notice for content such as defining subclasses, extending internal
interfaces, implementing repository-specific protocols, or following
developer-only implementation conventions. Apply it even when the section is
already grouped beneath a broader developer heading, so readers entering the
tutorial at that section can identify its audience immediately.

Do not use this notice merely because a user-facing section contains advanced
options or technical background. It is reserved for material that ordinary
users do not need in order to use the documented public interface.
