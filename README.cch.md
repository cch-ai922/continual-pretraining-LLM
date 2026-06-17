# README.cch — Korean quality filters and diversity axes

Session change log + how-to. Everything you need to run the new filters and the
audience × register sweep **without further help from Claude**. All commands are
PowerShell on Windows.

> **Windows console UTF-8 note.** Some scripts print Korean to stdout (bucket
> names like `합쇼체`, audience like `어린이`). PowerShell defaults to cp1252
> and will fail with `UnicodeEncodeError`. Fix once per shell session:
>
> ```powershell
> $env:PYTHONUTF8 = 1                    # preferred — Python uses UTF-8 for I/O
> # or, alternatively:
> chcp 65001                              # switch the console codepage to UTF-8
> ```

---

## Contents

- [1. What changed in this session](#1-what-changed-in-this-session)
- [2. New module — `register_consistency.py`](#2-new-module--register_consistencypy)
- [3. New module — `translationese_scorer.py`](#3-new-module--translationese_scorerpy)
- [4. Modified — `build_sft_blend.py` (Stage 5)](#4-modified--build_sft_blendpy-stage-5)
- [5. Modified — `build_preference_data.py` (Stage 6)](#5-modified--build_preference_datapy-stage-6)
- [6. Modified — `eval_report.py` (Stage 9)](#6-modified--eval_reportpy-stage-9)
- [7. Modified — `generate_grounded_ko.py` (Stage 5)](#7-modified--generate_grounded_kopy-stage-5)
- [8. End-to-end run order](#8-end-to-end-run-order)
- [9. Calibration workflow](#9-calibration-workflow)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. What changed in this session

Two **new** standalone modules:

| File | Purpose |
|---|---|
| [05_sft/register_consistency.py](05_sft/register_consistency.py) | Detect Korean 높임말 / 문체 consistency within a response (합쇼체 / 해요체 / 문어체 / 해체). Deterministic, no model. |
| [05_sft/translationese_scorer.py](05_sft/translationese_scorer.py) | Tier-1 heuristic 번역투 detector. 9 morphological features. Deterministic, no model. Optional calibration. |

Four **modified** scripts that consume the new modules:

| File | What was wired in |
|---|---|
| [05_sft/build_sft_blend.py](05_sft/build_sft_blend.py) | Both filters run on Korean pools **before** sampling, with per-pool drop reporting. Filters default **ON**. |
| [06_dpo/build_preference_data.py](06_dpo/build_preference_data.py) | Register consistency is now **signal #2** in the DPO `pick()` tuple (between language and format). |
| [09_eval/eval_report.py](09_eval/eval_report.py) | Per-row register score / dominant bucket / consistency flag; per-checkpoint mean score + mix rate + bucket profile reported on the Korean slice. |
| [05_sft/generate_grounded_ko.py](05_sft/generate_grounded_ko.py) | New `--register` and `--register-filter` flags (on top of the existing `--audience`). Post-filter drops outputs whose dominant register doesn't match the requested one. |

Every script ships with a `--selftest` flag. Run them all in one go any time:

```powershell
python 05_sft/register_consistency.py --selftest; `
python 05_sft/translationese_scorer.py --selftest; `
python 05_sft/build_sft_blend.py --selftest; `
python 06_dpo/build_preference_data.py --selftest; `
python 09_eval/eval_report.py --selftest; `
python 05_sft/generate_grounded_ko.py --selftest
```

All six must print `PASS ...`. If any fails, do not proceed.

---

## 2. New module — `register_consistency.py`

### What it does

Korean speech levels are morphologically marked at the sentence-final verb. A
response that mixes them within itself is a defect (the canonical chatbot failure
mode: half formal `합쇼체` + half polite `해요체`). This module classifies each
sentence and reports whether the response is single-register or mixed.

| Bucket | Markers |
|---|---|
| `합쇼체` | `-습니다`, `-습니까`, `-십시오`, `-십시다`, `-읍시다`, `-니다`, `-니까` |
| `해요체` | `-이에요`, `-예요`, `-세요`, `-아요/어요/여요`, `-거든요`, `-네요`, `-군요`, bare `-요` |
| `문어체` | sentence-final `-한다`, `-된다`, `-있다`, `-없다`, `-이다`, `-였다`, bare `-다` |
| `해체` | `-이야`, `-야`, `-잖아`, `-거든`, `-더라`, `-구나`, `-네`, `-지` |

Detection scrubs fenced code, inline code, and embedded direct-speech quotes
(`「…」`, `『…』`, `"…"`) so a quote's register doesn't pollute the wrapper's.
List items and short interjection forms (`네` / `응` / `아니에요` / `아니야`) are
handled specially.

### Public API

```python
from register_consistency import (
    classify_sentence,        # -> "합쇼체" | "해요체" | "문어체" | "해체" | "neutral"
    register_distribution,    # -> {"합쇼체":n, "해요체":n, "문어체":n, "해체":n, "neutral":n}
    consistency_score,        # -> (score, dominant_bucket, n_register_sents)
    is_consistent,            # -> bool, the gate
    matches_target_register,  # -> bool, dominant equals the target
)
```

`is_consistent(text, threshold=0.8, min_sents=2)` returns `True` if either:
1. The response has fewer than `min_sents` register-bearing sentences (not enough
   signal to judge — neutral / lists / very short replies pass), OR
2. The dominant bucket covers ≥ `threshold` of all register-bearing sentences.

### CLI

```powershell
python 05_sft/register_consistency.py --in <input.jsonl> [--out <output.jsonl>] `
    [--field {auto|text|assistant}] `
    [--threshold 0.8] [--min-sents 2] `
    [--target {합쇼체|해요체|문어체|해체}]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--in` | path | **required** | JSONL with either `{"text": "..."}` rows OR chat-format `{"messages":[…]}` rows. |
| `--out` | path | none | If given, write only the **kept** rows to this path. Without it, the script just prints stats. |
| `--field` | enum | `auto` | Where to find the scored text. `auto` picks `text` if present, else the assistant turn. |
| `--threshold` | float | `0.8` | Dominant-bucket fraction required to call a response consistent. Lower = more permissive. |
| `--min-sents` | int | `2` | Responses with fewer register-bearing sentences are kept unconditionally. |
| `--target` | enum | none | If given, also drop rows whose dominant bucket differs (use after a register-targeted generation pass). |

### Example

```powershell
# filter mixed-register rows out of translated SFT
python 05_sft/register_consistency.py `
    --in data/ko_sft_translated.jsonl `
    --out data/ko_sft_translated_consistent.jsonl

# enforce 해요체 on a register-targeted generation
python 05_sft/register_consistency.py `
    --in data/grounded_qa_general_haeyo.jsonl `
    --out data/grounded_qa_general_haeyo_clean.jsonl `
    --target 해요체
```

### Output format

```
total            : 50,000
consistent (kept): 41,123
all-neutral (kept): 3,891
mixed (dropped)  : 4,986
dominant-bucket counts (non-neutral rows):
  합쇼체: 18,402
  해요체: 21,840
  문어체: 881
  해체:   0
-> kept rows written to data/ko_sft_translated_consistent.jsonl
```

---

## 3. New module — `translationese_scorer.py`

### What it does

번역투 (translationese) is Korean text whose surface betrays it was translated
from English: overt pronouns where Korean would drop them, `에 의해` passive
calques, deep `의` modifier chains, `것이다` ending overuse, etc. This module
computes 9 surface features and flags a row if at least `--min-signals` of them
exceed their thresholds (default 3 of 9 — precision-friendly).

No model, no labels needed at runtime. Optional `--calibrate` recomputes
thresholds from your own native/translated reference data.

### The 9 features

| # | Name | What it counts | What translationese does |
|---|---|---|---|
| 1 | `pronoun_per_100c` | Overt 그/그녀/그것/그들/이것 + 은/는/이/가/을/를/의 per 100 chars | Native Korean uses zero-anaphora; translation copies English pronouns |
| 2 | `passive_per_100c` | `에 의해 / 에 의하여 / 에 의한` per 100 chars | Direct calque of English passive voice |
| 3 | `deul_per_100c` | `들` + particle per 100 chars | English `-s` forces plural marking Korean usually drops |
| 4 | `connective_per_sent` | `그리고/그러나/하지만/그래서/따라서/그러므로` at sentence start | Native Korean prefers conjunctive endings (`-고/-지만/-니까`) |
| 5 | `thing_end_ratio` | Sentences ending in `것이다 / 것입니다 / 것이에요` | Formal translated Korean overuses this nominalizer |
| 6 | `dem_dep_per_100c` | `이/그/저 + 것/곳/때/사람/등/들/이/점/분` per 100 chars | English the/this/that calque |
| 7 | `chain_deep_count` | NPs with three consecutive `의` joins | English NP nesting copied structurally |
| 8 | `chain_med_per_100c` | Two-consecutive `의` joins per 100 chars | Weaker version of #7 |
| 9 | `sent_len_mean` | Mean sentence length in chars | Translationese runs long |

### Public API

```python
from translationese_scorer import (
    features,            # -> dict of 9 feature values
    composite_score,     # -> (n_fired, [fired_names])
    is_translationese,   # -> bool, the gate
    calibrate,           # -> dict, recomputed thresholds
    DEFAULT_THRESHOLDS,  # the shipped defaults
    FEATURE_NAMES,
)
```

### CLI (filter mode)

```powershell
python 05_sft/translationese_scorer.py --in <input.jsonl> [--out <output.jsonl>] `
    [--field {auto|text|assistant}] `
    [--min-signals 3] `
    [--load-thresholds <th.json>]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--in` | path | **required** (unless calibrating) | JSONL with `{"text"}` or `{"messages"}` rows. |
| `--out` | path | none | If given, write rows that **did not fire** to this path. |
| `--field` | enum | `auto` | Same as register module. |
| `--min-signals` | int | `3` | Flag row as translationese if at least this many of the 9 features fire. Range 1..9. Lower = more aggressive. |
| `--load-thresholds` | path | none | Load per-feature thresholds from a JSON file produced by `--save-thresholds`. Overrides `DEFAULT_THRESHOLDS`. |

### CLI (calibration mode)

```powershell
python 05_sft/translationese_scorer.py --calibrate `
    --native <native_ko.jsonl> --translated <mt_ko.jsonl> `
    [--field {auto|text|assistant}] `
    --save-thresholds <th.json>
```

| Flag | Meaning |
|---|---|
| `--calibrate` | Switch to calibration mode. Does not write filtered output. |
| `--native` | JSONL of clean native Korean (~100-200 examples is enough). |
| `--translated` | JSONL of known MT Korean (same size). |
| `--save-thresholds` | JSON path to write the calibrated thresholds. Reload with `--load-thresholds`. |

Calibration sets each feature's threshold to the **p95** of the native
distribution. The console report shows per-feature separation:

```
feature                  native_p95   trans_median  separation
----------------------------------------------------------------
pronoun_per_100c             0.1234         3.4567  STRONG
passive_per_100c             0.0567         0.6789  STRONG
deul_per_100c                0.4321         0.5432  weak
...
```

- **STRONG** — translated median > 2× native p95 (good discriminator)
- **weak** — translated median between native p95 and 2× native p95
- **NONE** — translated median ≤ native p95 (feature doesn't discriminate on your data; consider raising `--min-signals` or dropping the feature in a fork)

### Examples

```powershell
# default thresholds, default min-signals=3
python 05_sft/translationese_scorer.py `
    --in data/ko_sft_translated.jsonl `
    --out data/ko_sft_translated_clean.jsonl

# calibrate on your own reference samples, then filter with those thresholds
python 05_sft/translationese_scorer.py --calibrate `
    --native data/native_ref.jsonl `
    --translated data/mt_ref.jsonl `
    --save-thresholds data/translationese_thresholds.json

python 05_sft/translationese_scorer.py `
    --in data/ko_sft_translated.jsonl `
    --out data/ko_sft_translated_clean.jsonl `
    --load-thresholds data/translationese_thresholds.json `
    --min-signals 2          # tighten after calibration
```

---

## 4. Modified — `build_sft_blend.py` (Stage 5)

### What's new

The original script only loaded the three pools, sampled, and shuffled. The new
version applies both quality filters to Korean pools **before** sampling, with
per-pool drop reporting split by reason.

Default behavior:
- **`translated_ko` pool**: both register and translationese filters applied.
- **`native_ko` pool**: register filter only. Translationese filter is OFF by
  default (genuine native idiosyncrasies look like 번역투 to the heuristic).
- **`english` pool**: never filtered (Korean checks don't apply).

### All CLI flags

```powershell
python 05_sft/build_sft_blend.py `
    --translated-ko <translated.jsonl> `
    [--native-ko <native.jsonl>] `
    [--english <en.jsonl>] `
    --out <blend.jsonl> `
    [--weights "0.70,0.20,0.10"] `
    [--total 300000] `
    [--seed 0] `
    [--no-filter-register] `
    [--no-filter-translationese] `
    [--filter-native-translationese] `
    [--register-threshold 0.8] `
    [--register-min-sents 2] `
    [--register-target {합쇼체|해요체|문어체|해체}] `
    [--translationese-min-signals 3]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--translated-ko` | path | **required** | JSONL of translated SFT (e.g. output of `translate_en_sft_to_ko.py`). |
| `--native-ko` | path | none | JSONL of grounded native SFT (output of `generate_grounded_ko.py`). |
| `--english` | path | none | JSONL of English SFT replay slice. |
| `--out` | path | **required** | Output blend JSONL. |
| `--weights` | csv | `0.70,0.20,0.10` | Sampling weights in order `translated, native, english`. Must sum to ~1.0. |
| `--total` | int | `300000` | Total examples to draw (with replacement). |
| `--seed` | int | `0` | RNG seed for reproducible mixes. |
| `--no-filter-register` | flag | OFF (filter ON) | Disable the 높임말/문체 consistency filter on **both** Korean pools. |
| `--no-filter-translationese` | flag | OFF (filter ON for `translated_ko`) | Disable the 번역투 filter on `translated_ko`. |
| `--filter-native-translationese` | flag | OFF | **Enable** the 번역투 filter on `native_ko` too (off by default to avoid dropping genuine native idiosyncrasies). |
| `--register-threshold` | float | `0.8` | Passed through to `is_consistent`. |
| `--register-min-sents` | int | `2` | Passed through to `is_consistent`. |
| `--register-target` | enum | none | If set, also drop Korean rows whose dominant register differs. |
| `--translationese-min-signals` | int | `3` | Passed through to `is_translationese`. |

### Example output

```
=== Korean quality filters ===
  translated_ko  in=  2,000,000  kept=  1,623,447  dropped=376,553 (18.8%)  [reg_mix=82,109 reg_target=0 trans=294,444]  applied=['register', 'translationese']
  native_ko      in=    500,000  kept=    487,201  dropped= 12,799 ( 2.6%)  [reg_mix=12,799 reg_target=0 trans=0]  applied=['register']
  english        in=    100,000  (English pool — Korean filters skipped)

=== blend ===
  translated_ko  weight=0.70  sampled=210,000  pool_size=1,623,447
  native_ko      weight=0.20  sampled=60,000  pool_size=487,201
  english        weight=0.10  sampled=30,000  pool_size=100,000

wrote 300,000 SFT examples -> data/ko_sft_blend.jsonl
```

The translated-pool drop rate **is the signal you read**:
- Drop rate **5-25%**: healthy, filters are catching real defects.
- Drop rate **< 2%**: filters aren't firing — translator is already very clean, or thresholds are too loose. Run `translationese_scorer.py --calibrate` to tighten.
- Drop rate **> 50%**: too aggressive. Either your translation is genuinely terrible (fix the translator), or thresholds are too tight. Loosen `--translationese-min-signals` from 3 → 4 first.

### Example commands

```powershell
# default: both filters on translated_ko, register-only on native_ko
python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated.jsonl `
    --native-ko    data/grounded_native.jsonl `
    --english      data/en_sft_slice.jsonl `
    --out          data/ko_sft_blend.jsonl

# enforce a target register across the whole SFT pool (all Korean rows must be 해요체)
python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated.jsonl `
    --native-ko    data/grounded_native_haeyo.jsonl `
    --out          data/ko_sft_blend_haeyo.jsonl `
    --register-target 해요체

# tighter translationese filter after calibration
python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated.jsonl `
    --out          data/ko_sft_blend.jsonl `
    --translationese-min-signals 2

# disable a filter (debugging)
python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated.jsonl `
    --out          data/ko_sft_blend_no_trans.jsonl `
    --no-filter-translationese
```

---

## 5. Modified — `build_preference_data.py` (Stage 6)

### What's new

The DPO `pick()` function now has **5 signals** ranked lexicographically (was 4):

| # | Signal | What it prefers |
|---|---|---|
| 1 | language | Korean response (korean_fraction ≥ 0.6) beats non-Korean |
| 2 | **register** | **NEW**: single-register response beats register-mixed |
| 3 | format | Followed requested format (JSON/list) beats not |
| 4 | faithfulness | For grounded prompts, faithful-to-source beats hallucinated |
| 5 | health | Non-degenerate (no repetition loops) beats degenerate |

Ranking is **lexicographic**: a higher-priority signal only hands the decision
to the next one if both candidates score equally. English responses pass the
register check trivially (no Korean register markers fire → `n_register_sents == 0`
→ `is_consistent` short-circuits to True), so signal #2 only differentiates
pairs where both candidates are Korean.

### No new CLI flags

Existing CLI is unchanged:

```powershell
python 06_dpo/build_preference_data.py --prompts <prompts.jsonl> --out <pairs.jsonl>
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--prompts` | path | **required** | JSONL with `{prompt, [format], [source]}` per row. |
| `--out` | path | **required** | Output DPO pairs JSONL with `{prompt, chosen, rejected}`. |

The `generate_pair(prompt)` hook still needs you to connect your SFT model
(vLLM/HF) — same as before.

### What this changes downstream

In Round-1 DPO training the rejected side will be **biased toward register-mixed
responses**. Expect the model after DPO to consolidate on a single register more
strongly. The default register the model gravitates to is whichever appears more
in the SFT pool (track this in `eval_report.py`'s bucket profile).

---

## 6. Modified — `eval_report.py` (Stage 9)

### What's new

Three additions to `score_rows`, one to `format_report`:

**Per-row** (added to each row dict):
- `_register_score` — consistency score (1.0 = perfectly single-register)
- `_register_dominant` — `"합쇼체" | "해요체" | "문어체" | "해체" | None`
- `_register_n_sents` — number of register-bearing sentences
- `_register_consistent` — bool, applying threshold + min_sents

**Per-slice aggregates** (added to `metrics["ko"]` and `metrics["en"]`):
- `n_register_judged` — count of rows with ≥ `register_min_sents` register-bearing sentences
- `mean_register_score` — average consistency score over judgable rows
- `register_mix_rate` — fraction of judgable rows where `_register_consistent == False`
- `register_buckets` — `{"합쇼체": n, "해요체": n, "문어체": n, "해체": n}` dominant-bucket counts

**Report line** (new line under the existing per-slice rows):

```
  Korean  : register: n_judged=412  mean_score=0.96  mix_rate=  4.4%  buckets[hapsyo=12%/haeyo=83%/muneo=5%/hae=0%]
```

(Bucket labels in the report are ASCII for cp1252 safety; they map 1:1 to the Korean buckets.)

### CLI

Two new flags on top of the existing ones:

```powershell
python 09_eval/eval_report.py --in <predictions.jsonl> `
    [--verifier {math|code|mcq|logic|format}] `
    [--min-korean 0.6] [--max-latin 0.15] `
    [--register-threshold 0.8] [--register-min-sents 2] `
    [--out <per_row_metrics.jsonl>]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--in` | path | **required** | Predictions JSONL (see schema in script docstring). |
| `--verifier` | enum | none | Default verifier name (math/code/mcq/logic/format). Per-row `verifier` field overrides. |
| `--min-korean` | float | `0.6` | Code-switch threshold (lower → more responses tagged switched). |
| `--max-latin` | float | `0.15` | Code-switch threshold (higher → more lenient on Latin script). |
| `--register-threshold` | float | `0.8` | Consistency threshold passed to scorer. |
| `--register-min-sents` | int | `2` | Min register-bearing sentences for a row to enter the aggregate. |
| `--out` | path | none | If given, write per-row enriched JSONL (includes all `_register_*` fields). |

### What to watch across checkpoints

This is the whole point of the new metrics:

1. **`mean_register_score`** trending **down** over checkpoints = your SFT/DPO pipeline is losing register discipline. Inspect the bucket profile.
2. **`register_mix_rate`** trending **up** = same problem, dual view.
3. **`register_buckets`** profile **shifting** between checkpoints = your model is changing its default register. Often a side effect of DPO when the SFT pool's register distribution differs from what DPO's verifier favors.
4. **`mean_register_score`** on the English slice should be `None` — English text never has Korean register markers so it's excluded from the aggregate. If it shows a number, something is mis-tagged.

### Example commands

```powershell
# basic — no verifier
python 09_eval/eval_report.py --in data/eval_predictions.jsonl

# math eval at a checkpoint
python 09_eval/eval_report.py --in data/eval_predictions.jsonl --verifier math

# stricter register gate (track tighter consistency)
python 09_eval/eval_report.py --in data/eval_predictions.jsonl `
    --register-threshold 0.9 --register-min-sents 3

# dump per-row metrics for offline analysis
python 09_eval/eval_report.py --in data/eval_predictions.jsonl `
    --out data/eval_per_row.jsonl
```

---

## 7. Modified — `generate_grounded_ko.py` (Stage 5)

### What's new

A second diversity axis (**register**) on top of the existing **audience** axis.
Both are Option A — *prompt-only* injection. No new exemplar files needed.

| Axis | Flag | Values | Default |
|---|---|---|---|
| audience | `--audience` | `어린이`, `중고급중학교생`, `일반`, `전문가`, `mixed` | `mixed` |
| register | `--register` | `합쇼체`, `해요체`, `문어체`, `mixed` | `mixed` |
| post-filter | `--register-filter` | flag | OFF |

`mixed` = no injection (legacy / original behavior). When non-mixed, an
instruction line is prepended to the few-shot prompt (steers the base model),
and a phrase is prepended to the SFT user message (teaches the trained model to
follow such requests at inference).

The two axes **stack**: with both set, the header reads
`다음 실례들을 참고하여, <audience instr> <register instr>` and the SFT user
message reads `<audience phrase> <register phrase> <task instruction> …`.

### Why register defaults to Option A (no new exemplars)

The shipped exemplar files (`exemplars_qa.json`, etc.) are written in one
register; rewriting them per-register is expensive (4 exemplars × 9 tasks ×
3 registers = 108 hand-authored exemplars). Option A steers the base model with
a single instruction line and uses `--register-filter` to drop misses. If the
miss rate is unacceptably high (> 50% on a given register after Stage 4 base
training), invest in register-specific exemplar files (Option B):
`exemplars_qa_haeyo.json`, `exemplars_reasoned_qa_hapsyo.json`, etc. Pass them
via `--exemplars` and the model gets a much stronger register signal.

### All CLI flags

```powershell
python 05_sft/generate_grounded_ko.py `
    --model <model_path_or_hf_id> `
    --passages <passages.jsonl> `
    --exemplars <exemplars_<task>.json> `
    --task {qa|reasoned_qa|multi_qa|summary|title|fact_extraction|explain_simply|outline|definition} `
    --out <out.jsonl> `
    [--audience {어린이|중고급중학교생|일반|전문가|mixed}] `
    [--register {합쇼체|해요체|문어체|mixed}] `
    [--register-filter] `
    [--min-korean 0.7] [--batch 256]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--model` | str | **required** | HF id or local path to your Stage-4 base model. |
| `--passages` | path | **required** | JSONL with `{"passage": "...", ["term": "..."]}` per row. `term` only used for `definition` task. |
| `--exemplars` | path | **required** | JSON file with 3-5 few-shot exemplars for the chosen task. Use `exemplars_<task>.json` or a register-specific variant (Option B). |
| `--task` | enum | `qa` | One of 9 task types. See script docstring for what each produces. |
| `--out` | path | **required** | Output JSONL of chat-format SFT examples. |
| `--audience` | enum | `mixed` | Audience instruction. `mixed` = no injection (legacy). |
| `--register` | enum | `mixed` | Register instruction. `mixed` = no injection. |
| `--register-filter` | flag | OFF | When `--register` is set (not `mixed`), drop outputs whose dominant register doesn't match. **Use this.** The drop count is your steering miss rate. |
| `--min-korean` | float | `0.7` | Drop generations whose `korean_fraction` < this. |
| `--batch` | int | `256` | Generation batch size. Adjust for your model's max batch on the GPU. |

### Sweep the cross-product from PowerShell

The script processes one `(task, audience, register)` combination per
invocation. Sweep with a nested loop:

```powershell
$audiences = "어린이","중고급중학교생","일반","전문가"
$registers = "합쇼체","해요체","문어체"

foreach ($aud in $audiences) {
  foreach ($reg in $registers) {
    $out = "data/grounded_qa_${aud}_${reg}.jsonl"
    python 05_sft/generate_grounded_ko.py `
        --model ./qwen3-ko-base-hf `
        --passages data/ko_passages.jsonl `
        --exemplars 05_sft/exemplars_qa.json `
        --task qa `
        --audience $aud --register $reg --register-filter `
        --out $out
  }
}
```

With 4 audiences × 3 registers = **12× the original per-passage data volume**
(before filters). Each combination's drop count is reported on its own line.

To sweep tasks too, wrap one more level:

```powershell
$tasks = "qa","reasoned_qa","multi_qa","summary","explain_simply"
foreach ($task in $tasks) {
  foreach ($aud in $audiences) {
    foreach ($reg in $registers) {
      $out = "data/grounded_${task}_${aud}_${reg}.jsonl"
      python 05_sft/generate_grounded_ko.py `
          --model ./qwen3-ko-base-hf `
          --passages data/ko_passages.jsonl `
          --exemplars "05_sft/exemplars_${task}.json" `
          --task $task `
          --audience $aud --register $reg --register-filter `
          --out $out
    }
  }
}
```

### Example output (per invocation)

```
task=qa audience=일반 register=해요체: kept 8,421 | dropped 1,579 (register off-target: 893) -> data/grounded_qa_일반_해요체.jsonl
```

Interpretation:
- `dropped` includes: parse failures, language-ID failures, faithfulness failures, register failures.
- `register off-target` is the **register steering miss rate** specifically. Computed when `--register-filter` is on.
- If `register off-target` is **> 50%** for a given register on your base model, it's evidence that prompt-only steering is insufficient and you should author Option-B register-specific exemplars for that register.

### Combining the sweep output

After the sweep, concatenate all JSONLs into one `native_ko.jsonl` for
`build_sft_blend.py`:

```powershell
Get-Content data/grounded_*.jsonl | Set-Content data/native_ko.jsonl
```

Then feed to `build_sft_blend.py` as `--native-ko`. The register filter in
`build_sft_blend.py` re-validates the data; with `--register-filter` already on
during generation, this should be a no-op for most rows.

---

## 8. End-to-end run order

After Stages 0-4 (base model trained), here is the Stage 5/6/9 flow with the new
filters wired in:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 5 — SFT data build                                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  English SFT (Alpaca etc.)                                               │
│         │                                                                │
│         ▼                                                                │
│  translate_en_sft_to_ko.py                                               │
│         │                                                                │
│         ▼                                                                │
│  ko_sft_translated.jsonl  ──────┐                                        │
│                                  │                                       │
│  ko_passages.jsonl                │                                       │
│         │                         │                                       │
│         ▼                         │                                       │
│  generate_grounded_ko.py           │                                      │
│  (loop over audience × register)   │                                      │
│  (--register-filter ON)             │                                     │
│         │                            │                                    │
│         ▼                             │                                   │
│  grounded_*.jsonl (12+ files)          │                                  │
│         │                               │                                 │
│         ▼ (concat)                       │                                │
│  native_ko.jsonl   ─────────────────────┤                                 │
│                                          ▼                                │
│                                build_sft_blend.py                         │
│                                (register + translationese filters,        │
│                                 per-pool drop stats)                      │
│                                          │                                │
│                                          ▼                                │
│                                ko_sft_blend.jsonl                         │
│                                          │                                │
│                                          ▼                                │
│                                    sft_train.py                           │
│                                          │                                │
│                                          ▼                                │
│                                ko_sft_model (HF)                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 6 — DPO                                                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  prompts.jsonl  ──►  build_preference_data.py                            │
│                     (pick() signal #2 = register)                        │
│                              │                                           │
│                              ▼                                           │
│                     ko_dpo_pairs.jsonl                                   │
│                              │                                           │
│                              ▼                                           │
│                          dpo_train.py                                    │
│                              │                                           │
│                              ▼                                           │
│                       ko_dpo_model (HF)                                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Stage 9 — eval at every checkpoint                                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  eval_set.jsonl  ──►  vLLM generate  ──►  predictions.jsonl              │
│                                                  │                       │
│                                                  ▼                       │
│                                          eval_report.py                  │
│                                          (Korean slice now reports       │
│                                           mean_register_score,           │
│                                           register_mix_rate,             │
│                                           register_buckets profile)      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Calibration workflow

The translationese scorer ships with sensible defaults that **work without
calibration**. Calibrate when you want to tighten precision/recall against your
specific translator's output.

### Step 1 — Assemble reference data

Two small JSONL files (~100-200 examples each is enough):

- `native_ref.jsonl` — clean native Korean. Source: a slice of your
  `ko_passages.jsonl` is fine (mostly clean by assumption).
- `mt_ref.jsonl` — known machine-translated Korean. Source: a sample of
  `ko_sft_translated.jsonl` (early-round output of `translate_en_sft_to_ko.py`).

Schema: each row is `{"text": "..."}` or chat format. The script auto-detects.

### Step 2 — Calibrate

```powershell
python 05_sft/translationese_scorer.py --calibrate `
    --native data/native_ref.jsonl `
    --translated data/mt_ref.jsonl `
    --save-thresholds data/translationese_thresholds.json
```

Read the separation report. Features with `NONE` separation are not
discriminating on your data — your translator doesn't make that mistake.
Features with `STRONG` separation are your money signals.

### Step 3 — Apply

```powershell
python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated.jsonl `
    --out data/ko_sft_blend.jsonl
```

build_sft_blend uses the shipped `DEFAULT_THRESHOLDS` directly (it imports
`is_translationese`). To use your calibrated thresholds, you either:

**(a)** filter once with the calibrated thresholds upstream, then pass the
filtered file as the blend input:

```powershell
python 05_sft/translationese_scorer.py `
    --in data/ko_sft_translated.jsonl `
    --out data/ko_sft_translated_clean.jsonl `
    --load-thresholds data/translationese_thresholds.json

python 05_sft/build_sft_blend.py `
    --translated-ko data/ko_sft_translated_clean.jsonl `
    --out data/ko_sft_blend.jsonl `
    --no-filter-translationese          # already filtered upstream
```

**(b)** or edit `DEFAULT_THRESHOLDS` in `translationese_scorer.py` to your
calibrated values (less flexible — only do this when calibration converges).

Recommend **(a)**.

### Step 4 — Iterate

Each SFT/DPO round, re-sample a fresh reference pair and re-calibrate. Your
model's translation behavior shifts, so the thresholds should track.

---

## 10. Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode characters` on Windows

PowerShell stdout is cp1252 by default. Set UTF-8 once per shell:

```powershell
$env:PYTHONUTF8 = 1
```

Or add to `$PROFILE`:

```powershell
Add-Content $PROFILE "`$env:PYTHONUTF8 = 1"
```

### `ImportError: No module named register_consistency` when running `build_sft_blend.py` from another directory

The sibling-import pattern in all four modified scripts uses
`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`. This should
work from any working directory. If it doesn't, you've probably moved one of
the files. Restore the original layout (`05_sft/register_consistency.py`,
`05_sft/translationese_scorer.py` siblings).

### `argparse.ArgumentError: invalid choice: '합쇼체'` when passing Korean as a CLI value

PowerShell argument parsing under cp1252 mangles UTF-8 strings before Python
sees them. Either:

```powershell
$env:PYTHONUTF8 = 1                              # fix at the Python level
# OR
chcp 65001                                       # fix at the console level
```

Verify with: `python -c "import sys; print(sys.argv)" --register 합쇼체` — the
last list element should be `'합쇼체'`.

### Translationese filter drops nothing on real data

Default `--min-signals 3` is precision-friendly. If your real data has lots of
번역투 that's escaping the filter:

1. Run `--calibrate` to tighten per-feature thresholds.
2. Lower `--min-signals` to 2.
3. If still too lenient, the heuristic features can't catch your specific
   translator's failure modes. Either swap to a Tier-2 LLM-judge filter (use
   your Round-1 SFT model as the judge) or fix the upstream translator.

### Register filter drops everything on a register-targeted generation

The base model isn't following the prompt-only register instruction. Three
options:

1. **Most likely cause:** Your Stage-4 base model has weak Korean and ignores
   the register instruction. Wait until later iterations (the SFT model is a
   stronger follower) or use Path C synthetic top-up.
2. Author Option-B register-specific exemplar files
   (`exemplars_qa_<register>.json`) and pass via `--exemplars`. Much stronger
   signal than prompt-only.
3. Run **without** `--register-filter` for now to harvest data, then post-filter
   with `register_consistency.py --target <register>` separately. Measure the
   miss rate and decide if it's investment-worthy.

### `consistency_score` returns `(1.0, None, 0)` on responses you expected to bucket

This means no sentence-final marker matched and the response has no
register-bearing sentences. Common causes:

- Bare `-아/-어/-여` verb endings ("맛있어", "재밌어") are intentionally
  **not** matched — they collide with noun-final position and would
  false-positive. Cost: 해체 declaratives written that way are scored as
  neutral. Documented limitation in the module docstring.
- Sentence not split because the period was inside a quote — the quote
  scrubbing removed it. Check your input.

### DPO pairs all tie on signal #2

If `pick()` returns `None` on a lot of pairs, both candidates are register-
consistent. That's not a bug — the model is producing consistent responses on
both temperatures, and you need other signals (faithfulness, format) to
discriminate. Make sure prompts carry `source` (for faithfulness) and `format`
(for format) hints when applicable.
