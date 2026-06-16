#!/usr/bin/env python3
"""
Stage 8 (verifier) — structured-output format verifier.

Teaches the model to produce well-formed structured output by rewarding format
adherence. Supports four common format types via a dispatch dict:

  json        : parse JSON cleanly; optionally check required keys / value types.
  list        : has at least N bulleted or numbered items.
  regex       : the (extracted or full) response matches a regex.
  md_table    : has a markdown table with the required column names + min rows.

The gold spec is a dict {"type": "json", ...} which lets a single training file
mix many format checks. Extraction is forgiving — JSON can be wrapped in a
```json``` block or in plain text; tables can have surrounding prose.
"""
import argparse, json, re, sys


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
_JSON_FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.S)


def extract_json(text: str):
    """Find a JSON object/array in the text. Returns the parsed Python object or
    None. Prefers fenced ```json``` blocks, then the largest balanced {...} or [...]."""
    if not text:
        return None
    for m in _JSON_FENCE.finditer(text):
        try:
            return json.loads(m.group(1))
        except Exception:
            continue
    # find balanced { ... } or [ ... ] candidates; try the longest first
    candidates = []
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0; start = -1
        for i, ch in enumerate(text):
            if ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start:i + 1])
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(cand)
        except Exception:
            continue
    try:                                          # last-ditch: maybe the whole response is JSON
        return json.loads(text.strip())
    except Exception:
        return None


_TYPE_MAP = {"str": str, "string": str, "int": int, "integer": int,
             "float": float, "number": (int, float), "bool": bool, "boolean": bool,
             "list": list, "array": list, "dict": dict, "object": dict, "null": type(None)}


def check_json_schema(obj, schema):
    """Tiny built-in schema checker (avoids the jsonschema dep). schema is:
        {"required": ["key1", "key2"], "types": {"key1": "str", "key2": "int"}}
    Both fields optional. Returns True iff every requirement holds."""
    if not isinstance(schema, dict):
        return True
    if not isinstance(obj, dict):
        return False
    for k in schema.get("required", []):
        if k not in obj:
            return False
    for k, typ in (schema.get("types") or {}).items():
        if k not in obj:
            continue
        T = _TYPE_MAP.get(typ.lower() if isinstance(typ, str) else None, typ)
        if not isinstance(obj[k], T):
            return False
    return True


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------
_LIST_LINE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")


def count_list_items(text: str) -> int:
    return 0 if not text else len(_LIST_LINE.findall(text))


# ---------------------------------------------------------------------------
# MARKDOWN TABLE
# ---------------------------------------------------------------------------
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)


def extract_md_table(text: str):
    """Return (headers: list[str], rows: list[list[str]]) for the FIRST table, or None."""
    if not text:
        return None
    sep = _TABLE_SEP.search(text)
    if not sep:
        return None
    lines = text[: sep.start()].splitlines()
    if not lines or "|" not in lines[-1]:
        return None
    headers = [c.strip() for c in lines[-1].strip().strip("|").split("|")]
    rows = []
    for line in text[sep.end():].splitlines():
        s = line.strip()
        if not s:
            if rows:
                break                              # blank line after rows ends the table
            continue                                # blank line right after separator: skip
        if "|" not in s:
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(cells)
        elif rows:
            break
    return headers, rows


# ---------------------------------------------------------------------------
# UNIFORM VERIFIER INTERFACE
# ---------------------------------------------------------------------------
def verify(model_output: str, gold) -> bool:
    """Dispatch on `gold['type']`. Examples of `gold`:

      {"type": "json"}                                 just parseable
      {"type": "json", "schema": {"required":["a"],
                                  "types":{"a":"int"}}}   parseable + schema
      {"type": "list", "min_items": 3}                 ≥3 bullets/numbered items
      {"type": "regex", "pattern": "^Answer:\\s*\\d+$"}  matches the regex
      {"type": "md_table", "columns": ["이름","나이"],
                           "min_rows": 2}              has those columns + ≥2 rows
    """
    if not isinstance(gold, dict):
        return False
    t = gold.get("type")
    if t == "json":
        obj = extract_json(model_output)
        if obj is None:
            return False
        return check_json_schema(obj, gold.get("schema"))
    if t == "list":
        return count_list_items(model_output) >= int(gold.get("min_items", 1))
    if t == "regex":
        flags = re.S | (re.I if gold.get("ignore_case") else 0) | (re.M if gold.get("multiline") else 0)
        return re.search(gold["pattern"], model_output or "", flags) is not None
    if t == "md_table":
        tbl = extract_md_table(model_output)
        if tbl is None:
            return False
        headers, rows = tbl
        cols = gold.get("columns") or []
        if cols and not all(c in headers for c in cols):
            return False
        if len(rows) < int(gold.get("min_rows", 1)):
            return False
        return True
    return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help='JSONL with {prediction, gold:{type:"json|list|regex|md_table", ...}}')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    n = ok = 0
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line); n += 1
        pred = r.get("prediction") or r.get("response") or r.get("output", "")
        v = verify(pred, r["gold"])
        ok += int(v)
        if fout:
            fout.write(json.dumps({**r, "verified": v}, ensure_ascii=False) + "\n")
    if fout:
        fout.close()
    print(f"verified {ok:,}/{n:,}  ({ok/max(n,1):.1%})")


# ---------------------------------------------------------------------------
def _selftest():
    # ---- JSON ----
    assert extract_json('Here you go: {"a": 1, "b": "x"} done.') == {"a": 1, "b": "x"}
    assert extract_json('```json\n{"k": [1, 2, 3]}\n```') == {"k": [1, 2, 3]}
    assert extract_json('garbage [1, 2, 3] more {"y": true}') == {"y": True}   # longest wins
    assert extract_json("no json here at all") is None
    # schema
    assert check_json_schema({"a": 1, "b": "x"}, {"required": ["a"], "types": {"a": "int"}})
    assert not check_json_schema({"a": "oops"}, {"types": {"a": "int"}})
    assert not check_json_schema({}, {"required": ["a"]})
    # via dispatch
    assert verify('{"name": "가", "age": 30}', {"type": "json",
        "schema": {"required": ["name", "age"], "types": {"age": "int"}}})
    assert not verify('not json', {"type": "json"})

    # ---- LIST ----
    txt = "Some intro.\n- one\n- two\n- three\nthe end"
    assert count_list_items(txt) == 3
    assert count_list_items("1. a\n2. b\n3) c\n") == 3
    assert count_list_items("• 항목 하나\n• 항목 둘") == 2
    assert verify(txt, {"type": "list", "min_items": 3})
    assert not verify(txt, {"type": "list", "min_items": 5})
    assert not verify("no list here", {"type": "list", "min_items": 1})

    # ---- REGEX ----
    assert verify("Answer: 42", {"type": "regex", "pattern": r"Answer:\s*\d+"})
    assert verify("answer: 42", {"type": "regex", "pattern": r"Answer:", "ignore_case": True})
    assert not verify("nope", {"type": "regex", "pattern": r"^\d+$"})

    # ---- MARKDOWN TABLE ----
    md = "intro\n\n| 이름 | 나이 |\n|-----|-----|\n| 철수 | 30 |\n| 영희 | 28 |\n\nafter"
    headers, rows = extract_md_table(md)
    assert headers == ["이름", "나이"]
    assert rows == [["철수", "30"], ["영희", "28"]]
    assert verify(md, {"type": "md_table", "columns": ["이름", "나이"], "min_rows": 2})
    assert not verify(md, {"type": "md_table", "columns": ["foo"], "min_rows": 1})
    assert not verify(md, {"type": "md_table", "columns": ["이름"], "min_rows": 5})
    assert extract_md_table("no table here") is None

    # bad gold
    assert not verify("anything", {"type": "unknown"})
    assert not verify("x", "not a dict")

    print("PASS all format-verifier tests (json + list + regex + md_table)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
