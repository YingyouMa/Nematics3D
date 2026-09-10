# Tutorial Writing Guide

This guide records the writing and formatting conventions for `Nematics3D`
tutorials. Apply these rules to new tutorials and to existing tutorials when
they are substantially revised.

## Two tutorial layers

`Nematics3D` deliberately keeps two complementary tutorial layers:

- `tutorials/guided/` is a progressive, task-driven introduction. It should
  teach a reader how to work with the package without first requiring them to
  understand the class hierarchy or memorize the API.
- `tutorials/reference/` is the toolbook layer. It documents individual
  functions, classes, methods, and reusable mechanisms in enough detail to be
  consulted independently.

Do not turn a guided tutorial into a shortened reference tutorial. A guided
tutorial should introduce an abstraction only when the reader needs that
abstraction to accomplish the current task. Detailed option lists, complete
method inventories, implementation protocols, and edge cases normally belong
in `tutorials/reference/`.

## Guided tutorial narrative

The current numbered notebooks in `tutorials/guided/` define the preferred
style. Preserve the following pattern when extending the sequence:

1. Begin from a concrete question or task a user naturally has.
2. Reuse objects and data from earlier tutorials when possible, so the series
   forms one continuous workflow rather than a collection of isolated demos.
3. Introduce one main new mental model at a time.
4. Show the shortest useful workflow before explaining deeper machinery.
5. Teach discovery mechanisms such as autocomplete, `show_...`, `act_...`,
   and object relations instead of asking readers to memorize large APIs.
6. Explicitly defer exhaustive detail to the corresponding reference
   tutorial.
7. End by connecting the current task to the next natural question when the
   notebook is part of the numbered guided sequence.

Prefer a task-oriented title and story over an implementation-oriented one.
For example, if `RegistryBase` is needed to explain how users find and reuse
objects, first motivate the user problem and introduce `RegistryBase` inside
that workflow. Do not make the inheritance hierarchy the reason a beginner is
expected to read a notebook.

### Guided tutorial names are questions

Every guided tutorial must be framed as a question in both its filename and
its top-level heading. In most cases, prefer a practical `How to ...?`
question because the guided layer is organized around tasks rather than API
topics.

Use other question forms such as `What ...?` or `Why ...?` when they express
the reader's actual question more naturally. Do not force `How to` onto a
conceptual tutorial merely for uniformity.

The notebook filename should mirror the heading in lowercase snake case,
keeping the numeric sequence prefix and omitting punctuation that is invalid
or awkward in filenames. For example:

```text
03_how_to_understand_an_unfamiliar_object.ipynb
# 03 — How to understand an unfamiliar object?
```

This convention is also a design check: if a proposed guided tutorial cannot
be expressed as a useful reader question, it is probably organized around an
implementation topic and should either be reframed as a task or moved to the
reference layer.

## API links in tutorial prose

When tutorial prose mentions a public `Nematics3D` function, class, or method,
link the identifier to its reference/toolbook notebook on its first meaningful
occurrence in that section. For example:

```markdown
[`quick_visualize_q()`](../reference/quick/quick_visualize_q.ipynb)
```

Adjust the relative path for the location of the notebook containing the
link. Keep the visible identifier in inline-code style.

The reference link is part of the documentation contract: guided prose says
*why and when* the API is useful, while the linked reference notebook says
*exactly how* to use it.

If a public API used by a tutorial does not yet have a reference notebook,
create the appropriate directory and a valid placeholder `.ipynb` at the
location where the eventual reference documentation belongs. Link to that
placeholder immediately rather than leaving an unresolved documentation debt.
The placeholder may contain only a title and a short note that the reference
tutorial is pending, but it must be valid notebook JSON so documentation tools
can open it.

Choose the reference path according to the package/API organization rather
than according to whichever guided tutorial first happened to use the API.
Methods belonging to a documented class should normally live under that
class's reference directory.

Do not create reference notebooks for ordinary Python syntax, local variables,
or external-library APIs merely because they appear in an example. External
software should link to its own documentation only when such a link materially
helps the tutorial.

## Practical completion standard

This guide is a collection of writing tools, not a checklist that every
tutorial must satisfy item by item. Applying every possible section and rule
to every function can create disproportionate work and make a tutorial longer
than its subject warrants. Select the sections that materially help readers,
and omit optional sections that have no useful content.

When judgment is needed, use the
[`q_diagonalize()` tutorial](../reference/q_field/diagonalization/q_diagonalize.ipynb) as the repository's
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
