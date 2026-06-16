#!/usr/bin/env python3
"""Generate Sudoku puzzles programmatically → {problem, gold} for verify_logic.

Sudoku puzzles are essentially free: no dataset download, no contamination concerns,
no licensing — generate as many as you want at the difficulty you want. We use
backtracking with shuffled digit order for diverse solutions, then blank N cells.

⚠️  We do NOT check for unique solvability — that requires a separate solver pass
and would slow generation 10x. For RL training data this is fine; the verifier
accepts ANY valid completion that respects the pre-filled cells, which is what
we want. If you need unique-solution puzzles for eval, run a solver post-hoc and
filter.

Difficulty preset → blanks:  easy=35  medium=45  hard=55
"""
import argparse, json, random, sys


def random_solved_grid(rng: random.Random) -> list[list[int]]:
    """Backtracking solver with shuffled digit order produces a random valid grid."""
    grid = [[0] * 9 for _ in range(9)]

    def fill(cell: int = 0) -> bool:
        if cell == 81:
            return True
        r, c = divmod(cell, 9)
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for d in digits:
            if any(grid[r][i] == d for i in range(9)): continue
            if any(grid[i][c] == d for i in range(9)): continue
            br, bc = (r // 3) * 3, (c // 3) * 3
            if any(grid[br + i][bc + j] == d for i in range(3) for j in range(3)): continue
            grid[r][c] = d
            if fill(cell + 1):
                return True
            grid[r][c] = 0
        return False

    fill()
    return grid


def make_puzzle(solved: list[list[int]], n_blanks: int, rng: random.Random) -> list[list[int]]:
    puzzle = [row[:] for row in solved]
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    for r, c in cells[:n_blanks]:
        puzzle[r][c] = 0
    return puzzle


def format_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join("." if x == 0 else str(x) for x in row) for row in grid)


PROBLEM_TEMPLATE = (
    "다음 sudoku 수수께끼를 푸시오. 매 행, 매 렬, 그리고 매 3x3 칸에는 "
    "1부터 9까지의 수자가 한번씩 들어가야 합니다. 점(.)은 빈칸을 나타냅니다.\n\n{grid}"
)
DIFFICULTY_BLANKS = {"easy": 35, "medium": 45, "hard": 55}


def generate_one(difficulty: str, rng: random.Random) -> dict:
    n_blanks = DIFFICULTY_BLANKS[difficulty]
    solved = random_solved_grid(rng)
    puzzle = make_puzzle(solved, n_blanks, rng)
    return {"problem": PROBLEM_TEMPLATE.format(grid=format_grid(puzzle)),
            "gold": {"type": "sudoku", "puzzle": puzzle}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="number of puzzles to generate")
    ap.add_argument("--difficulty", default="mixed",
                    choices=["easy", "medium", "hard", "mixed"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            d = args.difficulty if args.difficulty != "mixed" else rng.choice(["easy", "medium", "hard"])
            f.write(json.dumps(generate_one(d, rng), ensure_ascii=False) + "\n")
    print(f"wrote {args.n:,} puzzles ({args.difficulty}) -> {args.out}")


def _selftest():
    rng = random.Random(0)
    # solved grid is valid
    g = random_solved_grid(rng)
    assert all(set(row) == set(range(1, 10)) for row in g)
    for c in range(9):
        assert set(g[r][c] for r in range(9)) == set(range(1, 10))
    # puzzle preserves cells where not blanked
    p = make_puzzle(g, 30, rng)
    blanks = sum(1 for r in range(9) for c in range(9) if p[r][c] == 0)
    assert blanks == 30
    for r in range(9):
        for c in range(9):
            if p[r][c] != 0:
                assert p[r][c] == g[r][c]
    # full pipeline
    item = generate_one("easy", rng)
    assert item["gold"]["type"] == "sudoku"
    assert len(item["gold"]["puzzle"]) == 9
    assert "스도쿠" in item["problem"]
    # round-trip: format the solution back and verify_logic should accept it
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_logic import verify
    solution_text = "풀이:\n" + format_grid(g)
    fake_puzzle_from_solved = {"type": "sudoku", "puzzle": [r[:] for r in g]}
    assert verify(solution_text, fake_puzzle_from_solved)
    print("PASS sudoku generator (incl. round-trip through verify_logic)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
