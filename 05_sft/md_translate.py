"""
Markdown-preserving translation wrapper for a sentence-based MT system.

THE IDEA
--------
Your translator collapses whitespace, splits lines, and translates sentence by
sentence -- so it destroys any markdown you feed it. The fix is to never let it
see structure. This module:

  1. Walks the markdown LINE BY LINE and keeps all structure (fences, headings,
     list markers, blockquotes, table pipes, indentation) in Python -- it is
     never sent to the translator.
  2. For each piece of prose, it MASKS things that must survive verbatim
     (inline code, math, raw HTML, URLs) with rare-bracket sentinels, translates
     the naked prose, then restores them.
  3. Links and emphasis get their visible text translated while markers/URLs are
     preserved.

Plug your MT system into translate_text() and call translate_markdown(md).

ASSUMPTIONS / LIMITS (read these)
---------------------------------
* Assumes roughly one logical unit per line, which is how most model-generated
  markdown (and SFT responses) looks. If your source HARD-WRAPS paragraphs
  across multiple lines, merge soft-wrapped lines into single paragraphs first.
* Emphasis inner text is translated slightly out of sentence context, which can
  cost a little quality on the bolded phrase -- an acceptable trade for keeping
  the formatting.
* TEST the sentinel characters against YOUR translator once (see __main__). If
  it mangles 〔 〕, swap _SENT_OPEN/_SENT_CLOSE for tokens it passes through.
"""

import re

# --------------------------------------------------------------------------
# 1. PLUG YOUR TRANSLATOR HERE
# --------------------------------------------------------------------------
def translate_text(text: str) -> str:
    """Replace this body with a call to your sentence-based MT system.

    It will ONLY ever receive clean prose: no markdown structure, with inline
    code / math / html / urls already replaced by sentinel tokens that must be
    passed through unchanged.
    """
    raise NotImplementedError("Wire up your MT system here.")


# --------------------------------------------------------------------------
# 2. SENTINELS (rare brackets survive most NMT systems untouched)
# --------------------------------------------------------------------------
_SENT_OPEN, _SENT_CLOSE = "\u3014", "\u3015"  # 〔 〕

def _sent(i: int) -> str:
    return f"{_SENT_OPEN}{i}{_SENT_CLOSE}"

# tolerant: allows the translator to have inserted stray spaces inside
_SENT_RE = re.compile(rf"{_SENT_OPEN}\s*(\d+)\s*{_SENT_CLOSE}")


# --------------------------------------------------------------------------
# 3. INLINE PROTECTION
# --------------------------------------------------------------------------
# spans whose CONTENT must never be translated (stored raw, restored verbatim).
# Handled BEFORE links so they can't be confused with link syntax.
_RAW_PRE = [
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),       # images (must precede link rule)
    re.compile(r"`[^`]*`"),                    # inline code
    re.compile(r"\$\$[^\n]*?\$\$"),            # single-line display math $$...$$
    # inline math $...$ : opener not followed by space, closer not preceded by
    # space -> matches "$E=mc^2$" / "$a + b$" but NOT currency like "$5 ... $10".
    re.compile(r"\$(?!\s)[^$\n]+?(?<!\s)\$"),
    re.compile(r"\\\((?:.|\n)*?\\\)"),         # inline math \( ... \)
    re.compile(r"\\\[[^\n]*?\\\]"),            # single-line display math \[...\]
    re.compile(r"<[^>]+>"),                    # raw html tags
]
# bare urls: handled AFTER links, and stop at markdown punctuation so we don't
# swallow a link's closing ) or ].
_BARE_URL = re.compile(r"https?://[^\s)\]]+")

# emphasis: translate the inner words, keep the markers. double before single.
_EMPHASIS = [
    re.compile(r"\*\*(.+?)\*\*"),
    re.compile(r"__(.+?)__"),
    re.compile(r"\*(.+?)\*"),
    re.compile(r"(?<!\w)_(.+?)_(?!\w)"),
]
_EMPHASIS_WRAP = ["**", "__", "*", "_"]

# links: [label](url) -> translate label, keep url
_LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")


def _translate_prose(text: str) -> str:
    """Translate one prose fragment while preserving inline formatting."""
    if not text.strip():
        return text

    store = []

    def stash(s: str) -> str:
        store.append(s)
        return _sent(len(store) - 1)

    # 3a. raw inline spans (code/math/html/images) -> sentinel, never translated
    for pat in _RAW_PRE:
        text = pat.sub(lambda m: stash(m.group(0)), text)

    # 3b. links -> translate label, keep url, whole thing becomes a sentinel
    def link_repl(m):
        label, url = m.group(1), m.group(2)
        tlabel = _restore(translate_text(label), store) if label.strip() else label
        return stash(f"[{tlabel}]({url})")
    text = _LINK.sub(link_repl, text)

    # 3c. any remaining bare urls -> sentinel
    text = _BARE_URL.sub(lambda m: stash(m.group(0)), text)

    # 3d. emphasis -> translate inner, re-wrap, become a sentinel
    for pat, wrap in zip(_EMPHASIS, _EMPHASIS_WRAP):
        def emph_repl(m, w=wrap):
            inner = _restore(translate_text(m.group(1)), store)
            return stash(f"{w}{inner}{w}")
        text = pat.sub(emph_repl, text)

    # 3e. translate what's left (naked prose + sentinels)
    text = translate_text(text)

    # 3f. restore everything
    return _restore(text, store)


def _restore(text: str, store) -> str:
    # loop because stored strings may themselves contain sentinels
    prev = None
    while prev != text:
        prev = text
        text = _SENT_RE.sub(lambda m: store[int(m.group(1))], text)
    return text


# --------------------------------------------------------------------------
# 4. BLOCK-LEVEL DRIVER (structure stays in Python, never sent to MT)
# --------------------------------------------------------------------------
_FENCE = re.compile(r"(```+|~~~+)")
_HEADING = re.compile(r"(#{1,6}\s+)(.*)")
_BLOCKQUOTE = re.compile(r"((?:>\s?)+)(.*)")
_LIST = re.compile(r"([-*+]\s+|\d+[.)]\s+)(.*)")
_TABLE_SEP = re.compile(r"[\s|:\-]+")


def _is_table_sep(s: str) -> bool:
    """True only for genuine GFM separator rows like |---|:--:|---|."""
    return "|" in s and "-" in s and bool(_TABLE_SEP.fullmatch(s))


def _split_table_cells(s: str):
    """Split a table row on pipes, but NOT on pipes that are escaped (\\|) or
    inside inline code (`...`) or math ($...$ / $$...$$). This keeps math like
    $|x|$ intact inside a cell."""
    cells, buf = [], []
    i, n = 0, len(s)
    in_code = False
    in_math = None  # None | "$" | "$$"
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:          # escaped char, keep both
            buf.append(s[i:i + 2]); i += 2; continue
        if c == "`":
            in_code = not in_code; buf.append(c); i += 1; continue
        if not in_code and c == "$":
            if s[i:i + 2] == "$$":
                in_math = None if in_math == "$$" else ("$$" if in_math is None else in_math)
                buf.append("$$"); i += 2; continue
            in_math = None if in_math == "$" else ("$" if in_math is None else in_math)
            buf.append("$"); i += 1; continue
        if c == "|" and not in_code and in_math is None:
            cells.append("".join(buf)); buf = []; i += 1; continue
        buf.append(c); i += 1
    cells.append("".join(buf))
    return cells


def _translate_table_row(line: str) -> str:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]
    cells = [_translate_prose(c) if c.strip() else c
             for c in _split_table_cells(stripped)]
    return indent + "|".join(cells)

# multi-line LaTeX math environments to pass through untouched
_MATH_ENVS = (r"(?:equation|align|gather|multline|flalign|alignat|eqnarray|"
              r"array|cases|split|smallmatrix|[pbvBV]?matrix)")
_MATH_ENV_BEGIN = re.compile(r"\\begin\{(" + _MATH_ENVS + r"\*?)\}")


def translate_markdown(md: str) -> str:
    out, in_fence, fence_ch = [], False, ""
    math_mode = None  # None | "dollar" | "bracket" | ("env", name)
    lines = md.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # fenced code blocks: toggle and pass through untouched
        fm = _FENCE.match(stripped)
        if fm:
            if not in_fence:
                in_fence, fence_ch = True, fm.group(1)[0]
            elif stripped[0] == fence_ch:
                in_fence = False
            out.append(line); i += 1; continue
        if in_fence:
            out.append(line); i += 1; continue

        # multi-line LaTeX math: pass through untouched (like code fences).
        # Single-line $$...$$, \[...\], $...$ are handled inline in _translate_prose.
        if math_mode is None:
            if stripped.count("$$") % 2 == 1:            # opens $$ ... $$
                math_mode = "dollar"; out.append(line); i += 1; continue
            if "\\[" in stripped and "\\]" not in stripped:  # opens \[ ... \]
                math_mode = "bracket"; out.append(line); i += 1; continue
            be = _MATH_ENV_BEGIN.search(stripped)             # opens \begin{env}
            if be and ("\\end{%s}" % be.group(1)) not in stripped:
                math_mode = ("env", be.group(1)); out.append(line); i += 1; continue
        else:
            out.append(line)                              # inside math: verbatim
            if math_mode == "dollar" and stripped.count("$$") % 2 == 1:
                math_mode = None
            elif math_mode == "bracket" and "\\]" in stripped:
                math_mode = None
            elif isinstance(math_mode, tuple) and \
                    ("\\end{%s}" % math_mode[1]) in stripped:
                math_mode = None
            i += 1; continue

        if not stripped:                      # blank line
            out.append(line); i += 1; continue

        # genuine GFM table: current line is a header ONLY if the NEXT line is a
        # separator row. This stops math like $|x|$ from looking like a table.
        if ("|" in stripped and not stripped.startswith(">")
                and i + 1 < n and _is_table_sep(lines[i + 1].strip())):
            out.append(_translate_table_row(line))   # header
            out.append(lines[i + 1])                 # separator verbatim
            i += 2
            while (i < n and lines[i].strip() and "|" in lines[i]
                   and not _is_table_sep(lines[i].strip())):
                out.append(_translate_table_row(lines[i]))
                i += 1
            continue

        # heading
        h = _HEADING.match(stripped)
        if h:
            out.append(indent + h.group(1) + _translate_prose(h.group(2)))
            i += 1; continue

        # blockquote (keep '>' markers)
        b = _BLOCKQUOTE.match(stripped)
        if b:
            out.append(indent + b.group(1) + _translate_prose(b.group(2)))
            i += 1; continue

        # list item (keep marker + indentation -> nesting preserved)
        l = _LIST.match(stripped)
        if l:
            out.append(indent + l.group(1) + _translate_prose(l.group(2)))
            i += 1; continue

        # plain paragraph line
        out.append(indent + _translate_prose(stripped))
        i += 1

    return "\n".join(out)


# --------------------------------------------------------------------------
# 5. DEMO  (uses a fake translator so you can see structure survive)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Fake MT that UPPERCASES words and collapses spaces + splits lines,
    # mimicking your translator's bad habits, to prove structure survives.
    def _fake(text):
        text = re.sub(r"[ \t]+", " ", text).strip()
        return " ".join(w.upper() if w.isalpha() else w for w in text.split(" "))

    translate_text = _fake  # noqa: F811  (monkeypatch for the demo)

    sample = """# How to install

Use the `pip install foo` command. See the **official docs** at
[the website](https://example.com) for more.

1. First, open a terminal
2. Then run the script below:

```python
print("this code must NOT be translated")
```

| Step | Action      |
|------|-------------|
| 1    | open it     |
| 2    | run **it**  |

> Note: keep your *config file* safe.
"""
    print(translate_markdown(sample))
