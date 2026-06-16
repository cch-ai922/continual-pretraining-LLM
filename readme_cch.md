# readme_cch.md — Session change log

This file records what was decided and what was added/changed in the project
during the planning session, so the changes are reviewable in one place.

Session date: 2026-06-16

## Decisions confirmed

### 1. Tokenizer path: keep REPLACEMENT (no switch to extension)

You confirmed staying on the replacement path that the repo name (`qwen3-8b-fully-replace-ko`) implies. **No code change here**, but the budget consequence is real and needs attention before Stage 4:

| Item | Default replacement budget | Your situation |
|---|---|---|
| Stage 4b total tokens | 100B (replacement guide recommends 80–120B) | ✓ Fine |
| Korean weight | 45% → 45B KO seen | ✓ Fine |
| Required real KO supply | ≥ 12B for 45B at < 4 epochs | ⚠ You have ~10B → 4.5 epochs |

**Implication.** With 10B real Korean and the replacement path, the Korean source hits ~4.5 epochs at default weights, above the Muennighoff ~4-epoch ceiling. Three things to do *before* kicking off Stage 4b:

1. **Top up Korean supply with synthetic** — generate ~5–10B synthetic KO from an external Korean-capable teacher (Solar 10.7B Apache-2.0, Polyglot-Ko MIT, or Kanana under its license) grounded in your 10B real. This is the playbook's strategy borrowed for the replacement path. Mix synthetic alongside real (collapse-safe: never replace, always accumulate). The synthetic top-up needs adding a new source to [build_data_mix.py:18-22](00_data/build_data_mix.py#L18-L22) — a new entry `ko_synth` with its `_text_document` prefix, then put real KO + synthetic KO together under the 45% Korean weight.
2. **Or shrink the Stage 4b budget to ~70B** — Korean seen drops to ~31.5B (~3.15 epochs), within ceiling. Replacement under-trained risk; English/Chinese degradation possible because the random-init embeddings haven't settled fully.
3. **Or shrink the Korean weight to 40%** and bump English/parallel — keeps replacement budget intact but trains less Korean.

Recommended: option (1) — synthetic top-up. Path-C from the analysis. The Stage-7 teacher loop already shows the project knows how to produce/filter synthetic safely; this just pulls that capability upstream of Stage 4.

### 2–4. Code changes

See "What was added" below.

---

## What was added

### Q3 — Register-consistency detector — ALREADY PRESENT

[05_sft/register_consistency.py](05_sft/register_consistency.py) was **already in the project** from an earlier session. It's a thorough implementation:

- Four register buckets (합쇼체 / 해요체 / 문어체 / 해체) with full suffix tables
- Quote-stripping (`「」`, `『』`, `""`, `''`) and code-fence stripping so embedded direct-speech doesn't pollute the wrapping register
- List-item / bullet / numbered-line skip so register-neutral fragments don't count
- Short-form responses table (`네`, `예`, `응`, `그래` etc.)
- Priority-order matching: 거든요 / 네요 / 군요 stay in 해요체 even though they could superficially look like 해체
- Chat-format JSONL driver (auto-picks the assistant turn)
- Target-register matching for filtering off-register generations
- Full `--selftest` covering 4 pure registers, mixed-register cases, quote/code/list scrubbing, priority order, short forms

Use it three ways (all in the file's docstring):

```bash
# (a) SFT data filter
python 05_sft/register_consistency.py --in translated_sft.jsonl --out consistent.jsonl

# (b) DPO preference signal — call consistency_score(text) from build_preference_data
# (c) Eval metric — call register_distribution(text) per generation

python 05_sft/register_consistency.py --selftest
```

**No change needed.** Just be aware it exists and wire it in to:

- [05_sft/build_sft_blend.py](05_sft/build_sft_blend.py) as a drop filter
- [06_dpo/build_preference_data.py](06_dpo/build_preference_data.py) as a preference signal
- [09_eval/eval_report.py](09_eval/eval_report.py) as a register-consistency metric

### Q1 — Translationese (번역투) scorer — ALREADY PRESENT

[05_sft/translationese_scorer.py](05_sft/translationese_scorer.py) was **already in the project**. It implements Tier-1 heuristic detection with 9 features:

1. `pronoun_per_100c` — overt 그/그녀/그것/그들 density (native Korean drops via zero-anaphora)
2. `passive_per_100c` — `에 의해`, `에 의하여`, `에 의한` (English passive calque)
3. `deul_per_100c` — `들` plural-marker over-attachment
4. `connective_per_sent` — standalone 그리고/그러나/그래서 at sentence start
5. `thing_end_ratio` — sentences ending in `것이다 / 것입니다 / 것이에요`
6. `dem_dep_per_100c` — demonstrative + dependent noun (이 것, 그 곳, 저 때)
7. `chain_deep_count` — three-deep 의-chains (`친구의 부모의 집의 ...`)
8. `chain_med_per_100c` — two-deep 의-chain density
9. `sent_len_mean` — mean sentence length in chars

Composite gate: flag as 번역투 if ≥ `--min-signals` features fire (default 3 of 9, precision-biased).

Calibration mode included — pass `--calibrate --native native.jsonl --translated mt.jsonl` to compute p95-of-native thresholds and a per-feature separation report.

```bash
# Filter translated SFT before it enters the blend
python 05_sft/translationese_scorer.py \
    --in ko_sft_translated.jsonl --out ko_sft_clean.jsonl --min-signals 3

# Calibrate thresholds on ~200 hand-labeled pairs
python 05_sft/translationese_scorer.py --calibrate \
    --native native_ko.jsonl --translated mt_ko.jsonl \
    --save-thresholds thresholds.json

python 05_sft/translationese_scorer.py --selftest
```

**No change needed.** Wire it into the Stage-5 build flow between `translate_en_sft_to_ko.py` and `build_sft_blend.py`.

### Q4 — Register-variant exemplars for 4 high-value tasks — NEW (12 files, 48 exemplars)

The previous exemplar set was register-uniform (mostly 문어체) and could not drive a register sweep. Added **12 new exemplar files** covering 4 high-value tasks × 3 registers × 4 exemplars each:

| Task           | 합쇼체 (hapsyo)                                                                          | 해요체 (haeyo)                                                                          | 문어체 (muncheo)                                                                            |
|----------------|------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| qa             | [exemplars_qa_hapsyo.json](05_sft/exemplars_qa_hapsyo.json)                              | [exemplars_qa_haeyo.json](05_sft/exemplars_qa_haeyo.json)                               | [exemplars_qa_muncheo.json](05_sft/exemplars_qa_muncheo.json)                               |
| reasoned_qa    | [exemplars_reasoned_qa_hapsyo.json](05_sft/exemplars_reasoned_qa_hapsyo.json)            | [exemplars_reasoned_qa_haeyo.json](05_sft/exemplars_reasoned_qa_haeyo.json)             | [exemplars_reasoned_qa_muncheo.json](05_sft/exemplars_reasoned_qa_muncheo.json)             |
| multi_qa       | [exemplars_multi_qa_hapsyo.json](05_sft/exemplars_multi_qa_hapsyo.json)                  | [exemplars_multi_qa_haeyo.json](05_sft/exemplars_multi_qa_haeyo.json)                   | [exemplars_multi_qa_muncheo.json](05_sft/exemplars_multi_qa_muncheo.json)                   |
| explain_simply | [exemplars_explain_simply_hapsyo.json](05_sft/exemplars_explain_simply_hapsyo.json)      | [exemplars_explain_simply_haeyo.json](05_sft/exemplars_explain_simply_haeyo.json)       | [exemplars_explain_simply_muncheo.json](05_sft/exemplars_explain_simply_muncheo.json)       |

**Other 5 tasks** (`title`, `summary`, `fact_extraction`, `outline`, `definition`) intentionally **not** given register variants — they are mostly register-neutral (titles aren't conjugated, summaries default to 문어체) and the existing single-file exemplars are fine.

**No edit to `generate_grounded_ko.py`** was needed — it already accepts any `--exemplars` path. Use the new files like this:

```bash
# 합쇼체 (formal high) — for formal/news-style outputs
python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf \
    --passages ko_passages.jsonl \
    --exemplars 05_sft/exemplars_qa_hapsyo.json \
    --task qa --out qa_hapsyo.jsonl

# 해요체 (informal polite) — for chatbot/conversational outputs
python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf \
    --passages ko_passages.jsonl \
    --exemplars 05_sft/exemplars_qa_haeyo.json \
    --task qa --out qa_haeyo.jsonl

# 문어체 (literary/declarative) — for expository/article outputs
python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf \
    --passages ko_passages.jsonl \
    --exemplars 05_sft/exemplars_qa_muncheo.json \
    --task qa --out qa_muncheo.jsonl
```

**Quality control after generation.** Each generation should be passed through:

1. `register_consistency.matches_target_register(text, target)` — drop outputs whose dominant register doesn't match what was requested.
2. `translationese_scorer.is_translationese(text)` — drop translationese-shaped outputs.
3. Existing `faithfulness_scorer.py` + `korean_fraction` filters.

This is the full Stage-5 filter stack now achievable without humans.

**Suggested register sweep ratio** for a balanced SFT pool:

- 해요체 (haeyo): 60% — chatbot default
- 합쇼체 (hapsyo): 25% — formal contexts
- 문어체 (muncheo): 15% — expository/article contexts

(Skip 반말 entirely — chatbots almost never need it.)

### Q4 follow-up — Audience-axis injection in `generate_grounded_ko.py` — NEW

Adds the second half of the playbook's diversity grid: **audience × register**. Register comes from the exemplar file (Option B above); audience is injected as an instruction. The two axes sweep independently.

Three changes to [05_sft/generate_grounded_ko.py](05_sft/generate_grounded_ko.py), all **backwards-compatible** (defaults preserve legacy behavior, existing call sites still work without modification):

1. **`AUDIENCES` table at the top** — 4 entries (`어린이` / `중고급중학교생` / `일반` / `전문가`), each a `(few-shot-header instruction, user-message phrase)` pair.
2. **`build_prompt(..., audience_instr="")`** — when given, prepends `다음 실례들을 참고하여, {instruction}` as a single header line above the exemplar block. Base model sees one instruction line, then the few-shot exemplars, then the open prompt.
3. **`to_chat(..., audience_phrase="")`** — when given, prepends the short phrase to the user-message instruction. Trains the downstream SFT model to follow explicit audience requests at inference.
4. **`--audience {어린이, 중고급중학교생, 일반, 전문가, mixed}`** CLI flag — `mixed` (default) = no injection (legacy). Other values look up the pair in `AUDIENCES` and pass through both functions.
5. **Selftest extended** — covers (a) header injection + exemplar structure preserved, (b) user-message phrase injection, (c) empty-arg backwards compatibility, (d) 4-way distinctness across audiences, (e) non-qa task coverage.

Use it like this — stacks with the register-variant exemplars for a full diversity sweep:

```bash
# Sweep audience × register for the qa task
for register in hapsyo haeyo muncheo; do
  for audience in 어린이 중고급중학교생 일반 전문가; do
    python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf \
        --passages ko_passages.jsonl \
        --exemplars 05_sft/exemplars_qa_${register}.json \
        --audience ${audience} --task qa \
        --out qa_${register}_${audience}.jsonl
  done
done
```

3 registers × 4 audiences = **12 sweep slots per task per passage**, vs. 1 with the old single-exemplar / no-axis behavior.

**Honest limitation** (also in the file's docstring): the base model has weak Korean and cannot perfectly follow the audience instruction from a single prefix line — the few-shot exemplars don't themselves vary by audience. Expect best-effort adherence (~60-80% on a strong base, lower on a weak one). Spot-check outputs and, if needed, build a downstream audience-checker (vocabulary level / sentence length / specialized-term density) — the analogue of `register_consistency.py` for the audience axis. Register adherence is much stronger because the exemplars demonstrate it directly.

---

## Two issues flagged but NOT touched

These are real issues in the project. I did NOT modify any existing file to fix them; deciding whether to sweep is your call. Listing them here so they're not forgotten.

### Issue 1 — Train/eval orthography mismatch (DPRK corpus vs ROK benchmarks)

The project uses **DPRK orthography** uniformly throughout: 조선 / 조선어 / 우리글 / 조선반도 / 량반 / 형성되였다 / 세종왕 / 발전도상나라 / 조선전쟁. This includes both the 9 original exemplar files and the 12 register-variant exemplar files added in this session.

This is internally consistent but creates a **train/eval mismatch**:

- Eval benchmarks (KMMLU / KoBEST / HAERAE / CLIcK) are all **ROK** — they test ROK spellings, ROK historiographical terms (한국전쟁 not 조선전쟁), ROK cultural-linguistic knowledge.
- A DPRK-orthography-trained model evaluated on ROK benchmarks will under-score on vocabulary recall and orthography even when the underlying knowledge is correct.

If you want to align with the ROK eval target, the sweep would touch:

- All 21 exemplar files in `05_sft/` (the 9 originals + the 12 register variants)
- [00_data/build_parallel_jsonl.py:21](00_data/build_parallel_jsonl.py#L21) — `조선어로 번역하시오` → `한국어로 번역하시오`
- Discussions in [REVIEW_REPORT.md](REVIEW_REPORT.md) and [DATA_MIX_AND_ORDERING.md](DATA_MIX_AND_ORDERING.md) — historical / annotation only
- Plus a careful pass through the source Korean monolingual corpus before Stage 4b, since that's where the bulk of the orthography signal comes from

A mechanical sed sweep won't be safe — `조선` legitimately appears when discussing the Joseon dynasty (`조선시대`, `조선왕조`) and must stay there even on a full ROK sweep. Needs a careful pass with whitelisted contexts.

**If you stay on DPRK**, this isn't a code defect — it's a known cost. Document it as a limitation in the project README and either (a) accept the eval under-score or (b) build a DPRK-side eval set (no public benchmark exists today; would need authoring).

### Issue 2 — `faithfulness_scorer.py` uses the BASE model for NLI

[05_sft/faithfulness_scorer.py](05_sft/faithfulness_scorer.py) does a few-shot NLI judgment with the base model (compares `logP(' 예')` vs `logP(' 아니오')`). Base models without NLI fine-tuning are mediocre entailment judges. The PMI signal in the same file is more reliable. Calibration with ~150 hand-labeled generations (as the docstring says) is essential before trusting the NLI score.

Not changed. Just flagging.

---

## Summary of files

**Created in this session (12 files):**

```text
05_sft/exemplars_qa_hapsyo.json
05_sft/exemplars_qa_haeyo.json
05_sft/exemplars_qa_muncheo.json
05_sft/exemplars_reasoned_qa_hapsyo.json
05_sft/exemplars_reasoned_qa_haeyo.json
05_sft/exemplars_reasoned_qa_muncheo.json
05_sft/exemplars_multi_qa_hapsyo.json
05_sft/exemplars_multi_qa_haeyo.json
05_sft/exemplars_multi_qa_muncheo.json
05_sft/exemplars_explain_simply_hapsyo.json
05_sft/exemplars_explain_simply_haeyo.json
05_sft/exemplars_explain_simply_muncheo.json
```

**Already present (no rewrite needed):**

```text
05_sft/register_consistency.py     (Q3 — 높임말/문체 consistency detector)
05_sft/translationese_scorer.py    (Q1 — 번역투 Tier-1 heuristic filter)
```

**Modified in this session (1 file):**

```text
05_sft/generate_grounded_ko.py     (added AUDIENCES table + --audience flag +
                                    optional injection params on build_prompt and
                                    to_chat; defaults preserve legacy behavior;
                                    selftest extended with 5 new assertions)
```

**Existing files not modified:**

```text
00_data/build_data_mix.py          (would need ko_synth source entry for Path-C synthetic top-up)
all 9 original exemplars_*.json    (preserved as-is)
```

---

## Next actions you may want to take

1. **Pick a Korean teacher** for synthetic top-up (Solar 10.7B / Polyglot-Ko / Kanana). Apache-2.0 / MIT licenses preferred.
2. **Author/source ~10B synthetic KO** grounded in your 10B real, using the teacher.
3. **Add `ko_synth_text_document` to [build_data_mix.py:18-22](00_data/build_data_mix.py#L18-L22)** as a sibling of the existing `korean` entry; split the 45% Korean weight across real+synth.
4. **Decide on the DPRK↔ROK orthography sweep** before any data generation runs. Doing it after Stage 4 means re-training.
5. **Calibrate `translationese_scorer.py` thresholds** on ~200 hand-labeled (native, MT) pairs from your data.
6. **Calibrate `faithfulness_scorer.py` thresholds** on ~150 hand-labeled grounded generations.
7. **Wire `register_consistency` into Stages 5 / 6 / 9** as described above.
