# Annotation Guidelines — Korean SFT Quality Filters Paper

For native Korean annotators. Three annotation tasks: **번역투 (translationese)**,
**문체 (speech-level consistency)**, **audience-match + pairwise preference**.

Each task has its own form. Calibrate on a 40-item pilot before scaling.

---

## General principles

- **Read each item completely before scoring.** No skimming.
- **Score only what you see.** Do not assume context that isn't in the item.
- **Flag, don't guess.** If you can't decide, use the "uncertain / can't tell"
  option, not a coin-flip rating.
- **Privacy.** Do NOT search the web for any item. Treat each as in-distribution.
- **Confidentiality.** Items, ratings, and lab procedures are confidential.
- **Pay & time.** Hourly rate plus quality bonus (calibrated to your blind-gold
  agreement Q-score). Pace yourself; quality matters more than throughput.

---

## Task 1 — Translationese (번역투) annotation

### What is 번역투?
Korean text whose *surface* betrays it was translated from a different language
(usually English), even though the words are individually correct. Hallmarks:

1. **Over-use of pronouns** where native Korean drops them.
   - Translationese: "그는 학교에 갔다. 그는 거기서 친구를 만났다."
   - Native: "학교에 갔다. 거기서 친구를 만났다."
2. **English passive-shaped Korean** with `에 의해`, `에 의하여`, `되어지다`.
   - Translationese: "이 결정은 위원회에 의해 내려졌다."
   - Native: "위원회가 이 결정을 내렸다."
3. **`-의` noun-phrase chains** that mirror English `the X of Y of Z`.
   - Translationese: "친구의 부모의 집의 문" → very awkward
   - Native: "친구네 부모님 집의 문" or split into clauses
4. **`것이다` sentence endings** over-used to mimic English copular structure.
   - Translationese: "이것은 사실인 것이다."
   - Native: "이건 사실이다."
5. **Standalone connectives** at sentence start (`그리고`, `그러나`, `따라서`) where
   native Korean prefers conjunctive endings (`-고`, `-지만`, `-니까`).
6. **Verbose `이/그/저 + 의존명사`** that calque English `this/that/the X`.
7. **Long sentences** where native Korean would split into multiple clauses.

### What is NOT 번역투?
- Formal academic Korean is *not* 번역투 by itself, even though it shares some
  surface features (passive-voice, 것이다 endings, long sentences). The
  question is whether the surface *feels* like Korean grammar or English-shaped.
- Sino-Korean vocabulary is not 번역투 (e.g., 환경 / 분석 / 결과 are perfectly
  native technical Korean).
- A skilled human translator's output may not be 번역투 — they preserve native
  Korean rhythm. The target of this filter is *sloppy machine-translation*.

### Scoring rubric (3 options)

| Score | Label | Meaning |
|---|---|---|
| **0** | clearly 번역투 | The text reads as English-shaped Korean. Native speaker would not write this. |
| **1** | borderline / uncertain | Has some 번역투 features but a native could plausibly write this. |
| **2** | clearly native | Reads as natural Korean. No giveaways of being translated. |

### Calibration items (pilot of 40 — these will be scored by 3 annotators
independently for κ calculation)
Items are sampled across:
- 10 native Korean Wikipedia paragraphs
- 10 native Korean news articles (Hankyoreh, Chosun, JoongAng)
- 10 Korean text produced by Helsinki-NLP/opus-mt-en-ko (translation of EN Wiki)
- 10 Korean text from KAIST parallel corpus (human-translated side)

After the pilot, compute κ across all annotator pairs. Target: κ ≥ 0.70. If
below, refine these guidelines (specifically, the borderline category) and
re-pilot.

### Common annotation mistakes to avoid

- **Bias toward formal Korean = 번역투.** Formal Korean uses 것이다 and passives
  legitimately. Score based on *whether the surface is English-shaped*, not on
  formality.
- **Bias toward conversational Korean = native.** A formal academic paragraph in
  perfectly natural Korean should still score 2 (clearly native).
- **Topic-vocabulary bias.** Don't penalize technical vocabulary as
  translationese — a native Korean STEM paper uses 환경 / 분석 / 결과 freely.
- **Length bias.** Long sentences are not automatically 번역투 — check the syntax.

---

## Task 2 — Speech-level (문체) annotation

### What are the Korean speech-level buckets?

| Bucket | Romanization | Sentence-final markers | Use |
|---|---|---|---|
| **합쇼체** | hapsyo | -습니다, -습니까, -십시오 | formal high — news, business, formal speech |
| **해요체** | haeyo | -요, -아요/어요, -이에요/예요, -세요 | informal polite — most chatbots, polite conversation |
| **문어체** | muncheo | sentence-final -다 (declarative) — 한다, 이다, 있다 | literary, expository writing, narration |
| **해체 / 반말** | banmal | -아/어, -야, -지, -네 | informal low — friends, intimate |

### Task 2a — Per-sentence classification (1,000 items)

For each single sentence, pick ONE bucket:

- 합쇼체
- 해요체  
- 문어체
- 해체 (반말)
- unclassifiable / fragment / list-item

**Edge cases:**
- Quoted speech inside a wrapping sentence: classify the *wrapping* sentence's
  register, not the quote's.
- List items (`- 첫째`) and bare phrase fragments: mark as "unclassifiable".
- Single-word responses (`네.`, `그래.`): mark as "unclassifiable" unless
  clearly polite (`네` → 해요체) vs informal (`그래` → 해체).
- 학교에 갔다. — bare -다 in narrative context = 문어체.
- 학교에 갔다. (spoken response to a friend) = 해체. Use context if available;
  default to 문어체 if unclear.

### Task 2b — Per-response consistency labeling (500 items)

For each multi-sentence response (2–10 sentences), pick ONE:

- **Consistent** — all sentences in the same register bucket (small slips are OK
  — 1 mixed sentence in 6 still counts as consistent if dominant bucket > 80%)
- **Mixed** — clear mixing across registers (e.g., starts in 합쇼체, drops to
  해요체 mid-response). This is the *defect* the filter targets.

### Calibration items (40 pilot)
- 8 pure 합쇼체 responses
- 8 pure 해요체
- 8 pure 문어체
- 4 pure 해체
- 8 deliberately mixed (the failure mode we care about)
- 4 ambiguous (single-sentence, register-neutral fragments)

Target κ on the per-sentence labels: ≥ 0.80. On per-response consistency: ≥ 0.75.

---

## Task 3 — Audience-match annotation (Set C, 2,400 items)

### What's the audience axis?

The grounded-generation script can request output for a target audience:

- **어린이** — elementary-school child. Vocabulary should be simple, sentences
  short, Sino-Korean vocabulary minimal, examples concrete.
- **중고급중학교생** — middle/high school student. Vocabulary at standard textbook
  level, sentence complexity moderate.
- **일반** — general adult reader. Default. Mid-range vocabulary, normal sentence
  complexity.
- **전문가** — domain expert. Technical vocabulary expected, dense sentences,
  precision over accessibility.

### Scoring rubric (5-point Likert per output)

For each generated output, given the (passage, requested audience):

| Score | Meaning |
|---|---|
| **5** | Clearly written for the target audience. Vocabulary, complexity, and style match. |
| **4** | Mostly for the target audience with minor mismatches. |
| **3** | Could apply to the target audience but indistinguishable from "일반" / generic. |
| **2** | Mismatched — clearly written for a different audience. |
| **1** | Wildly mismatched (e.g., requested "어린이" but output uses graduate-level vocabulary). |

**Adherence rate** = % of outputs rated ≥ 4.

### Don't penalize for content faithfulness here.
This task is about *audience match*, not factual accuracy. A perfectly-targeted
어린이 explanation that hallucinated facts still rates high on audience-match
(faithfulness is annotated separately in Set E).

### Calibration items (40 pilot)
- 10 outputs requested as 어린이 — actually simple, short
- 10 outputs requested as 어린이 — actually adult-level (failure cases)
- 10 outputs requested as 전문가 — actually technical
- 10 outputs requested as 전문가 — actually generic/일반-level

Target κ: ≥ 0.70.

---

## Task 4 — Pairwise preference (Set D, 600 pairs)

### Format

You see TWO responses to the same prompt, labeled "A" and "B". You don't know
which model produced which.

### Scoring rubric (5-point)

| Score | Meaning |
|---|---|
| **+2** | Strongly prefer A |
| **+1** | Mildly prefer A |
| **0** | Tie / no preference |
| **-1** | Mildly prefer B |
| **-2** | Strongly prefer B |

### What to consider

In order of priority:
1. **Faithfulness / correctness** — does the response correctly address the
   prompt? Hallucinations and refusals lose to faithful responses.
2. **Naturalness of Korean** — does it read as native Korean? 번역투 loses.
3. **Register appropriateness** — does the response match the prompt's
   register? A 반말 response to a formal question loses.
4. **Helpfulness** — is the response actually useful?
5. **Style consistency** — does it stay in one register throughout?

### Things to deliberately ignore
- Response length (longer ≠ better)
- Whether A or B "sounds smarter" — substance only
- Personal preference for verbosity / brevity
- Specific opinions in the response (unless wrong-as-fact)

### Calibration items (20 pilot pairs)
- 5 clear A-wins (V4 output vs deliberately broken V1)
- 5 clear B-wins (V1 vs broken V4)
- 5 tie pairs (two paraphrases of the same V1 output)
- 5 ambiguous

Target κ on the 5-point bucketing: ≥ 0.55. Pairwise is intrinsically harder than
per-bucket; lower κ target is realistic.

---

## Adjudication procedure

When 3 annotators disagree:

### Per-bucket tasks (Tasks 1, 2)
- If 2 of 3 agree → majority wins
- If all 3 different → escalate to lead annotator for binding decision

### 5-point Likert (Tasks 3, 4)
- If the bucketed labels span ≥ 2 buckets apart → escalate to lead annotator
- If within 1 bucket → average

### Blind-gold sanity checks
- 10% of every batch is re-rated by the lead annotator (who pre-rated a 100-item
  gold set)
- Annotator quality Q = % agreement with blind gold on the 10% sample
- Q ≥ 0.90 → full pay + bonus
- 0.85 ≤ Q < 0.90 → recalibration (review guidelines, re-rate flagged items)
- Q < 0.85 → pause batch, retraining session, restart

---

## Logistics

### Tools
- Annotation can be done in a spreadsheet (Google Sheets / Excel) or a
  lightweight annotation tool (Doccano, Prodigy, label-studio). For 5–6k items
  with simple rubrics, Google Sheets is fastest to set up.
- One row per item. Columns: `item_id, set, task, item_text (read-only), label,
  notes, time_spent`.
- Time per item:
  - Task 1 (per-text): ~1 minute
  - Task 2a (per-sentence): ~30 seconds
  - Task 2b (per-response): ~1 minute
  - Task 3 (5-point per-output): ~1.5 minutes
  - Task 4 (pairwise): ~2 minutes

### Workflow
1. Lead annotator pre-rates the 100-item blind-gold set (once, ~6 hours)
2. Recruit 3+ annotators; run 40-item pilot per task
3. Compute κ on pilot. Refine guidelines if needed.
4. Run production batches of 200 items. Compute Q on the 10% blind gold.
5. Pay per batch, with bonus tied to Q.
6. Adjudicate disagreements as they arise (don't queue them; resolve same week).

### Communication
- Weekly check-in (30 min) for question resolution
- Slack/Discord channel for real-time Q&A
- Document common edge cases as they emerge → append to these guidelines and
  share with all annotators

---

## Ethics & welfare

- Cap continuous annotation sessions at 90 minutes (eye-strain, decision-fatigue)
- Rotate annotators between tasks (don't have one person do 6 hours of
  pairwise-preference back-to-back)
- Do not require translation-quality judgment from annotators who haven't
  studied or worked in Korean→English translation; calibrate accordingly
- Annotators may flag items they find offensive / triggering and skip them; the
  lead re-assigns
- Cover annotator names in any published artifact (use IDs only)
- Pay above the legal floor (Korean minimum wage 10,320 KRW/hr in 2026)

---

## Output format

For each labeled set, produce a single CSV/JSONL with these columns minimum:

```
item_id, item_text, [optional context], annotator_id, label, notes, time_spent_sec, batch_id
```

Aggregated (one row per item) for analysis:
```
item_id, gold_label, annotator_1, annotator_2, annotator_3, majority, disagreement
```

These will be released (annotator IDs anonymized) as the paper's supplementary
materials.
