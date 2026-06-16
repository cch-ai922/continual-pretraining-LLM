# Source → `{problem, gold}` Converters

Each script converts one upstream source to the uniform schema `rejection_sample.py`
consumes: JSONL with `{"problem": str, "gold": <verifier-specific>}`. Run any
converter with `--selftest` first to verify the transformation logic before pointing
it at real data.

## Quick reference

| Converter           | Verifier | Source (HF id)                            | Schema notes                                       |
|---------------------|----------|-------------------------------------------|----------------------------------------------------|
| `from_gsm8k.py`     | math     | `gsm8k` (config `main`)                   | gold = number after `####`, commas stripped       |
| `from_math.py`      | math     | `hendrycks/competition_math`              | gold = LAST `\boxed{...}` content, balanced braces |
| `from_big_math.py`  | math     | `SynthLabsAI/Big-Math-RL-Verified`        | answer field already extracted; rename only        |
| `from_humaneval.py` | code     | `openai_humaneval`                        | gold = `{tests + check(entry_point), timeout}`     |
| `from_mbpp.py`      | code     | `mbpp` (config `sanitized`)               | gold = `{setup + test_list joined, timeout}`       |
| `from_mmlu.py`      | mcq      | `cais/mmlu` (subset `all` or per-subject) | answer index → letter; emits `n_options`           |
| `from_kmmlu.py`      | mcq      | `HAERAE-HUB/KMMLU` (subset = subject)     | A/B/C/D columns or options list; 1-based answers   |
| `gen_sudoku.py`     | logic    | (programmatic, no download)               | gold = `{"type":"sudoku","puzzle":[[...]]}`        |
| `gen_format.py`     | format   | (programmatic, no download)               | gold = `{"type":"json|list|md_table|regex",...}`   |

## Typical workflow

```bash
# 1. Install dependencies
pip install "datasets>=3.0"

# 2. Convert a source to {problem, gold} JSONL
python from_gsm8k.py --split train --out gsm8k_train.jsonl
python from_humaneval.py --out humaneval.jsonl
python from_mmlu.py --subset all --split test --out mmlu_test.jsonl
python from_kmmlu.py --subset Korean --lang hi --out kmmlu_ko.jsonl
python gen_sudoku.py --n 5000 --difficulty mixed --out sudoku_5k.jsonl
python gen_format.py --n 2000 --out ko_format_2k.jsonl

# 3. (optional, for math/code/mcq from English sources) translate the problem field
python ../translate_problems.py --in gsm8k_train.jsonl --out gsm8k_train_ko.jsonl
# verify_math is language-independent so `gold` passes through unchanged.

# 4. Run rejection sampling against the model
python ../rejection_sample.py --model ./qwen3-ko-sft-hf \
    --problems gsm8k_train_ko.jsonl --out-sft gsm8k_cot.jsonl --verifier math
```

## Per-source notes

### `from_gsm8k.py`
GSM8K's answer field has a chain-of-thought followed by `#### <number>`. The
converter strips the CoT and any commas in the number. Test split has ~1.3k
problems; train has 7.5k. Use train for RFT data, test for eval.

### `from_math.py`
The MATH solution field contains a LaTeX walkthrough ending in `\boxed{ANSWER}`.
The converter finds the LAST `\boxed{...}` (some solutions have intermediate boxes)
and handles nested braces (e.g. `\boxed{\frac{1}{2}}`). Difficulty levels are in
the `level` field if you want to filter (Level 1-5).

### `from_big_math.py`
Big-Math is the cleanest large math RL pool: deduped, filtered, single-numeric-
answer. The `answer` is already extracted, so this converter just renames the
fields. Recommended as the primary RL pool.

### `from_humaneval.py`
HumanEval's `prompt` is a partial function (`def f(...):` + docstring). The
converter wraps it in a code-fence and asks the model to complete it. The gold
`tests` field concatenates the dataset's `test` (which defines `check`) with a
final `check(entry_point)` call. Round-trip-tested through `verify_code`.

### `from_mbpp.py`
MBPP has a natural-language `text` description and a `test_list`. The converter
prepends a sample test to the problem (to clarify input/output expectations) and
joins all tests for the gold. Use the `sanitized` config (974 problems) — it's
the cleaned version; the `full` config (1k) has noisier prompts.

### `from_mmlu.py`
MMLU's `answer` is a 0-indexed integer into `choices`. The converter renders the
question with `A) ... B) ...` labels and outputs the corresponding letter. Adds
an `n_options` field (always 4 for MMLU, but kept consistent with KMMLU).

### `from_kmmlu.py`
The most valuable Korean source — natively Korean, Korea-centric (the Korean analogue of
MMLU, across 45 subjects). Field names have
shifted across releases, so the converter accepts multiple variants (`question`/
`prompt`/`query`; `options`/`choices`; `answer`/`correct_answer`/`label`).
**If the converter's selftest passes but no rows come out of your downloaded
dataset, your version uses different field names** — adjust the `FIELD_*`
constants at the top of the file. The lang filter (`--lang hi`) gates on
`language`/`lang` if those fields exist in your version.

### `gen_sudoku.py`
No download, no contamination, infinite supply. Difficulty preset controls the
number of blanks: easy=35, medium=45, hard=55. Does NOT check unique solvability
— that requires a second solver pass (10x slower). For RL training this is fine:
the verifier accepts any valid completion respecting the pre-filled cells. For
eval where you want unique-solution puzzles, post-filter with a Sudoku solver.

### `gen_format.py`
Synthetic Korean prompts asking for structured output. Four format types
matching `verify_format.py`: JSON (with schema), bullet/numbered list (with
min_items), markdown table (with required columns), and regex match. Templates
fill from a small Korean entity pool. The `--types` flag lets you generate a
domain-balanced training set. The selftest verifies a constructed correct
response passes each format check (round-trip with `verify_format`).

## Adding a new converter

Pattern: write `convert_row(row) -> {"problem": ..., "gold": ...} | None`
returning None for rows to skip, plus a `main()` that iterates `load_dataset(...)`
and a `_selftest()` that exercises `convert_row` with one synthetic example. Keep
the transformation logic separate from I/O so the selftest doesn't need network.
For converters whose gold has a non-trivial structure (code tests, sudoku puzzle),
add a round-trip test: construct a known-good model output and assert
`verify_<name>.verify(output, gold)` returns True. That catches schema mismatches
the unit tests would miss.
