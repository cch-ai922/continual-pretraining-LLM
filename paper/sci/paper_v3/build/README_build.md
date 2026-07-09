# Compiling `paper_full.md` (portable / self-contained)

This `build/` directory is a **self-contained bundle**: `paper_full.md` and its
`references.bib` sit side by side, and the document's YAML header declares
`bibliography: references.bib`.

Pandoc resolves that bare filename against the **current working directory**, not
the document's directory. So compile in one of these two ways:

**A. From this `build/` directory (simplest):**

```
cd paper_v3/build
pandoc --citeproc paper_full.md -o paper_full.pdf --pdf-engine=xelatex
```

**B. From any other directory — point the resource path at this folder:**

```
pandoc --citeproc --resource-path=path/to/paper_v3/build \
       path/to/paper_v3/build/paper_full.md -o paper_full.pdf --pdf-engine=xelatex
```

Both resolve all citations with **zero "citation not found" warnings** and produce
a fully populated References section. The canonical build
(`scripts/build_paper_v3.py`) also passes `--bibliography=<abs path>` explicitly,
so it works regardless of CWD; this note is for third-party / venue compiles that
use the document metadata alone.
