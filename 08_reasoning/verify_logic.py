#!/usr/bin/env python3
"""
Stage 8 (verifier) — logic-puzzle / constraint-satisfaction verifier.

Logic puzzles are too diverse for one canonical extractor, so this module gives:

  1. A CONCRETE built-in: Sudoku — extract a 9x9 grid from the response and check
     row/column/box uniqueness + consistency with the puzzle's pre-filled cells.

  2. A GENERIC framework: `verify_constraints(assignment, constraints)` where
     `constraints` is a list of callables `(assignment) -> bool`. Useful for
     scheduling, seating, assignment, satisfiability — anything you can express
     as "the answer is a mapping that satisfies these predicates."

To add a new puzzle type, write an `extract_<puzzle>` function (parse the model's
output into your puzzle's solution shape) and a list of constraints, then call
`verify_constraints(extracted, constraints)`.
"""
import argparse, json, re, sys


# ---------------------------------------------------------------------------
# SUDOKU
# ---------------------------------------------------------------------------
def extract_sudoku_grid(text: str, size: int = 9):
    """Find a sudoku grid in the model's response. Returns 9x9 list-of-lists of
    ints (0 means blank), or None. Tolerates separators like | - + . _."""
    if not text:
        return None
    rows = []
    for line in text.splitlines():
        # pull out only digits and dots/underscores (placeholders for blanks)
        cells = re.findall(r"[1-9]|[.0_]", line)
        if len(cells) == size:
            rows.append([int(c) if c.isdigit() and c != "0" else 0 for c in cells])
            if len(rows) == size:
                return rows
        elif rows and len(cells) != size:
            # consecutive same-size rows expected; resetting on noise is too eager,
            # so we tolerate the occasional malformed line by skipping it.
            continue
    return None if len(rows) != size else rows


def is_valid_sudoku(grid) -> bool:
    """Check 1-9 uniqueness across all rows, columns, and 3x3 boxes."""
    if not grid or len(grid) != 9 or any(len(r) != 9 for r in grid):
        return False
    target = set(range(1, 10))
    for row in grid:
        if set(row) != target:
            return False
    for c in range(9):
        if set(grid[r][c] for r in range(9)) != target:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {grid[br + i][bc + j] for i in range(3) for j in range(3)}
            if box != target:
                return False
    return True


def verify_sudoku(model_output: str, puzzle) -> bool:
    """Check the model's grid is (a) a valid sudoku, (b) consistent with the
    pre-filled cells in `puzzle` (a 9x9 grid where 0 = blank)."""
    grid = extract_sudoku_grid(model_output)
    if grid is None or not is_valid_sudoku(grid):
        return False
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] != 0 and grid[r][c] != puzzle[r][c]:
                return False
    return True


# ---------------------------------------------------------------------------
# GENERIC CONSTRAINT FRAMEWORK
# ---------------------------------------------------------------------------
def verify_constraints(assignment, constraints) -> bool:
    """`assignment`: any object representing the model's proposed solution.
    `constraints`: list of callables (assignment) -> bool. ALL must be True."""
    return all(c(assignment) for c in constraints)


def extract_kv_assignment(text: str, keys):
    """Parse 'key: value' (or 'key = value') lines from text. Returns dict on
    success (all keys found), else None. Tolerates Korean colons (：) and bullets."""
    if not text:
        return None
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*[-*•]?\s*([^:=]+?)\s*[:=：]\s*(.+?)\s*$", line)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip().rstrip(".,;。")
            if k in keys and k not in out:
                out[k] = v
    return out if set(out) >= set(keys) else None


_SAFE_BUILTINS = {
    "len": len, "set": set, "list": list, "dict": dict, "tuple": tuple,
    "sum": sum, "min": min, "max": max, "all": all, "any": any, "abs": abs,
    "int": int, "float": float, "str": str, "sorted": sorted, "range": range,
    "zip": zip, "enumerate": enumerate, "True": True, "False": False, "None": None,
}


# ---------------------------------------------------------------------------
# UNIFORM VERIFIER INTERFACE
# ---------------------------------------------------------------------------
def verify(model_output: str, gold) -> bool:
    """Dispatch on `gold['type']`:
       - 'sudoku'      : gold['puzzle']  (9x9 list-of-lists, 0 = blank)
       - 'assignment'  : gold['keys'] + gold['constraints'] (list of code strings
                         evaluated as predicates; UNSAFE, prefer in-code use of
                         verify_constraints with real callables instead)
    """
    if not isinstance(gold, dict):
        return False
    t = gold.get("type")
    if t == "sudoku":
        return verify_sudoku(model_output, gold["puzzle"])
    if t == "assignment":
        a = extract_kv_assignment(model_output, gold["keys"])
        if a is None:
            return False
        for src in gold.get("constraints", []):
            try:
                if not eval(src, {"__builtins__": _SAFE_BUILTINS}, {"a": a}):
                    return False
            except Exception:
                return False
        return True
    return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help='JSONL with {prediction, gold:{type:"sudoku|assignment", ...}}')
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
    # ---- sudoku ----
    solved = [
        [5,3,4,6,7,8,9,1,2],
        [6,7,2,1,9,5,3,4,8],
        [1,9,8,3,4,2,5,6,7],
        [8,5,9,7,6,1,4,2,3],
        [4,2,6,8,5,3,7,9,1],
        [7,1,3,9,2,4,8,5,6],
        [9,6,1,5,3,7,2,8,4],
        [2,8,7,4,1,9,6,3,5],
        [3,4,5,2,8,6,1,7,9],
    ]
    assert is_valid_sudoku(solved)
    bad = [r[:] for r in solved]; bad[0][0] = bad[0][1]
    assert not is_valid_sudoku(bad)
    # extract from a formatted response
    response = "다음은 풀이입니다:\n" + "\n".join(" ".join(str(x) for x in r) for r in solved)
    g = extract_sudoku_grid(response)
    assert g == solved
    # pipe-separated also works
    pipe = "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in solved)
    assert extract_sudoku_grid(pipe) == solved
    # blanks shown as . survive (and become 0s)
    puzzle = [r[:] for r in solved]; puzzle[0][0] = 0
    puzz_text = "\n".join(" ".join("." if x == 0 else str(x) for x in r) for r in puzzle)
    g2 = extract_sudoku_grid(puzz_text)
    assert g2 is not None and g2[0][0] == 0
    # consistency with puzzle (blank cell can be anything; pre-filled must match)
    p = [[0]*9 for _ in range(9)]; p[0][0] = 5
    assert verify_sudoku(response, p)
    p[0][0] = 6      # mismatch with the solution
    assert not verify_sudoku(response, p)
    # via dispatch
    assert verify(response, {"type": "sudoku", "puzzle": [[0]*9 for _ in range(9)]})

    # ---- generic constraint framework ----
    # toy: assign 3 days to 3 tasks, no task twice
    keys = ["task1", "task2", "task3"]
    text = "task1: Monday\ntask2: Wednesday\ntask3: Friday\n"
    a = extract_kv_assignment(text, keys)
    assert a == {"task1": "Monday", "task2": "Wednesday", "task3": "Friday"}
    no_dup = lambda a: len(set(a.values())) == len(a)
    has_friday = lambda a: "Friday" in a.values()
    assert verify_constraints(a, [no_dup, has_friday])
    assert not verify_constraints({"task1":"Mon","task2":"Mon","task3":"Fri"}, [no_dup])
    # Korean colons + bullets in the response
    ko_text = "• task1： 월요일\n• task2： 수요일\n• task3： 금요일"
    assert extract_kv_assignment(ko_text, keys) == {
        "task1": "월요일", "task2": "수요일", "task3": "금요일"}
    # dispatch with constraint strings (toy)
    assert verify(text, {"type": "assignment", "keys": keys,
                         "constraints": ["len(set(a.values())) == 3",
                                         "'Friday' in a.values()"]})
    assert not verify(text, {"type": "assignment", "keys": keys,
                             "constraints": ["'Saturday' in a.values()"]})

    print("PASS all logic-puzzle verifier tests (sudoku + generic constraints)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
