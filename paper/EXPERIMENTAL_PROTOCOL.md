# Experimental Protocol — Option A Paper

**Paper working title:** *Linguistically-Grounded Automatic Filters for Korean SFT Data:
Translationese, Speech-Level Consistency, and a Register × Audience Diversity Grid*

This document specifies every experiment, dataset, metric, and statistical-test
that the paper requires. Follow it end-to-end; the paper draft slots the numbers
your runs produce into clearly-marked placeholders.

---

## 0. Hardware, software, base model

### Hardware
- **2 nodes × 8 × H200** (16× H200, 141GB HBM/GPU, ~2.2TB aggregate)
- One node is the minimum required for the SFT runs (TP=2, DP=4 fits 8B at seq 4096); two nodes for faster turnaround.

### Software
| Component | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| PyTorch | 2.4+ | bf16-enabled build |
| transformers | 4.51+ | required for Qwen3 |
| vLLM | 0.6+ | used for generation + eval; SamplingParams(top_p=0.9, temperature=0.7) |
| TRL | 0.12+ | used for SFT — full FT, not LoRA, for cleanest filter signal |
| accelerate | 0.34+ | |
| datasets | 3.0+ | |
| **Frozen this paper** | git commit `8ff51d7` | initial commit of the code |

Pin everything (`pip freeze > requirements.lock.txt`) before E4 starts.

### Base model
**Use `Qwen/Qwen3-8B` directly as the SFT starting point.** Skip Stages 1–4
of the pipeline for this paper. The paper is about *filters*, not about base
adaptation. Holding the base constant isolates the filter contribution.

(Stages 1–4 are a separate paper — Option B in the recommendation. Do not mix
them.)

### Seeds and reproducibility
Three seeds per condition: 42, 1337, 2024.
Report mean ± std for every benchmark number.
A run is "valid" only if its training loss matches the seed-mean within 3σ.

---

## 1. Data preparation

### 1.1 Korean monolingual passages (for grounded generation in E3–E5)
| Source | Quantity | License | Notes |
|---|---|---|---|
| Korean Wikipedia dump (2025-01) | 50,000 paragraphs | CC-BY-SA | sample uniformly across topics |
| AI Hub `한국어 문어체 분야별 데이터` | 30,000 paragraphs | research-only | for grounded `summary`/`fact_extraction` |
| KLUE-corpus (training split) | 20,000 paragraphs | CC-BY-SA | for `qa`/`multi_qa` |

Pre-process: dedupe by 5-gram overlap, drop paragraphs <100 chars or >2000 chars.
Hash each paragraph, ensure no eval document leaks in (E1–E6 use `ko_passages.jsonl`;
E5 eval pool comes from a separate held-out 5,000-paragraph slice).

### 1.2 Translated EN→KO SFT base corpus (for E4 baseline)
| Source | Quantity | License | Notes |
|---|---|---|---|
| Alpaca-cleaned (English) | 50,000 examples | CC-BY-NC | translate to KO via Helsinki-NLP/opus-mt-en-ko |
| OpenOrca subset | 50,000 examples | MIT | translate via same MT system |
| FLAN-style task templates (EN) | 20,000 examples | Apache-2.0 | translate |

**Total: 120k translated KO SFT examples.** This is your V1 (baseline) SFT pool.

Translation system: **Helsinki-NLP/opus-mt-en-ko** (deterministic, reproducible,
free). Do NOT use commercial APIs (DeepL/Papago/GPT-4) — those are not
reproducible and licensing varies.

### 1.3 Grounded native Korean SFT (for E4 native-data slice)
Generate via `05_sft/generate_grounded_ko.py` from Qwen3-8B-base (the base
itself; see §0). Schedule:

| Task | Register file | Audience | Passages | Output examples |
|---|---|---|---|---|
| qa | exemplars_qa_haeyo.json | mixed | 5,000 | ~3,000 (~60% kept after filters) |
| qa | exemplars_qa_hapsyo.json | mixed | 5,000 | ~3,000 |
| qa | exemplars_qa_muncheo.json | mixed | 5,000 | ~3,000 |
| reasoned_qa | exemplars_reasoned_qa_haeyo.json | mixed | 3,000 | ~1,800 |
| multi_qa | exemplars_multi_qa_haeyo.json | mixed | 3,000 | ~1,800 |
| explain_simply | exemplars_explain_simply_haeyo.json | mixed | 2,000 | ~1,200 |
| summary / title / fact_extraction / outline / definition | unified exemplars | mixed | 1,000 each | ~600 each |

**Approx. ~15k native grounded examples**, register-mixed.

For E5 (diversity grid), generate the same 5,000 qa-task passages × {3 registers} × {4 audiences} = **60,000 sweep outputs**.

### 1.4 Hand-labeled evaluation sets

#### Set A — Translationese (for E1)
- 500 native Korean texts (sample from Korean Wikipedia, native blogs / news /
  literary corpora — confirm with a Korean linguistics PhD that they are clearly
  native)
- 500 machine-translated Korean texts (translate English Wikipedia & news →
  Korean via Helsinki opus-mt; quality-check the MT system actually produced
  Korean output for each item)
- 200 high-quality human-translated Korean texts (from KAIST or AI Hub parallel
  corpus; the HT side)
- **Total: 1,200 items labeled {native, MT, HT}**

Split: 30% calibration (360 items), 70% test (840 items). Stratified by source.

#### Set B — Speech-level register (for E2)
- 1,000 single-sentence labels: 200 × 합쇼체, 200 × 해요체, 200 × 문어체, 200 × 해체,
  200 × "unclassifiable / list / fragment"
- 500 multi-sentence response labels: 250 × "consistent (one register)", 250 ×
  "mixed (≥2 registers)"
- **Source:** sample from native Korean dialogue / news / literary corpora; verify
  with native annotator
- **Annotator agreement target:** Cohen's κ ≥ 0.85 on pilot batch (40 items)
  before scaling to the full set. If κ < 0.70 after retraining, sentence ambiguity
  is too high — refine guidelines.

#### Set C — Audience adherence (for E3)
- 200 source passages → 200 × 4 audience settings × 3 register settings = 2,400
  generated outputs to rate
- Each output rated by 3 native annotators on `audience_match` (1–5 Likert)
- Reduce to: was the audience clearly evident? (≥4 = yes)

#### Set D — End-to-end human pairwise (for E6)
- 200 prompts from a mix of:
  - 50 Ko-Alpaca-eval-style prompts
  - 50 KMMLU-style question prompts
  - 50 conversational prompts (custom-authored by native annotators)
  - 50 register-conditional prompts ("...합쇼체로 답하시오", "...해요체로 답하시오")
- Pairwise V4-vs-V1 + V4-vs-V2 + V4-vs-V3 = 600 pairwise comparisons
- Each pair rated by 3 annotators (blind, randomized order)

#### Set E — Faithfulness (for E4 ablation of `faithfulness_scorer.py`)
- 300 (passage, generated answer) pairs labeled "faithful / not-faithful" by 2
  annotators

**Total human annotation budget:** ~2,400 + 1,500 + 1,200 + 600 + 300 = **~6,000
labeled items**. At ~2 minutes/item × 3 annotators × 60% overlap = ~600
person-hours. Plan ≥ 3 native Korean annotators on a 2-week schedule.

### 1.5 Eval benchmarks (for E4 downstream)
| Benchmark | Subset | Why |
|---|---|---|
| **KMMLU** | All 45 subjects | broad knowledge MCQ |
| **KoBEST** | 5 tasks (BoolQ, COPA, HellaSwag, SentiNeg, WiC) | sentence-level NLU |
| **HAE-RAE** | Korean-cultural sub-tasks | cultural / linguistic |
| **CLIcK** | All | cultural / linguistic supplementary |
| **MMLU** (English regression) | 40 random subjects | catastrophic-forgetting alarm |
| **GSM8K** (English regression) | All test | math reasoning preserved? |

Use **lm-evaluation-harness** (lighteval is an alternative). Pin to a specific
commit. Report 0-shot accuracy unless the benchmark's standard is otherwise.

---

## 2. Experiments

### E1 — Translationese detector intrinsic evaluation
**Goal:** quantify how well the 9-feature heuristic detector classifies Korean
text as native vs MT-Korean.

**Procedure:**
1. Calibrate thresholds on 360-item calibration split (Set A): 
   ```
   python 05_sft/translationese_scorer.py --calibrate \
       --native native_cal.jsonl --translated mt_cal.jsonl \
       --save-thresholds thresholds.json
   ```
2. Evaluate on 840-item test split:
   - For each item, compute features and composite flag
   - Compute confusion: {true-native, true-MT/HT, predicted-native, predicted-MT}
3. Sweep `--min-signals` from 1 to 9 → ROC curve
4. **Baselines to compare:**
   - **B1 — Random**: pick "translationese" at corpus base rate
   - **B2 — KoBERT zero-shot**: prompt KoBERT (KR-BERT or klue/bert-base) with
     "이 문장은 번역체입니까? 예/아니오" → use yes/no logit ratio. (This is weak;
     it's the right baseline to demonstrate the value of structured heuristics.)
   - **B3 — GPT-4 5-shot judge**: prompt GPT-4 with 5 (native, MT) examples then
     judge unseen items. (Strong baseline; pay attention to costs ~$50.)
   - **B4 — Fine-tuned KoBERT classifier**: train KoBERT on the 360-item
     calibration set as a 2-class classifier; evaluate on test. (Upper bound; the
     heuristic is "deterministic, 0-data, license-free" — it should not beat
     this but should be competitive.)
5. **Per-feature ablation**: leave-one-out — re-calibrate thresholds, re-eval.
   For each removed feature, report ΔF1.

**Metrics:** Precision, Recall, F1 (macro), AUROC, calibration curve, per-feature
contribution.

**Outputs:** Table 3 in paper. Figure 1 (ROC curves). Figure 2 (per-feature LOO).

**Expected ranges** (placeholder targets — replace with measured):
- Heuristic F1: **0.78 – 0.86**
- GPT-4 5-shot F1: **0.85 – 0.92**
- Fine-tuned KoBERT F1: **0.90 – 0.95**
- KoBERT zero-shot F1: **0.55 – 0.65**
- Top-3 features by ΔF1: `thing_end_ratio`, `pronoun_per_100c`, `chain_deep_count`

### E2 — Speech-level consistency detector intrinsic evaluation
**Goal:** quantify per-sentence register classification and per-response
consistency detection.

**Procedure:**
1. Per-sentence: run `classify_sentence()` from `register_consistency.py` on
   the 1,000-sentence Set B. Compute confusion matrix across 4 buckets + neutral.
2. Per-response: run `consistency_score()` on the 500-response set. Threshold at
   0.8 (default); also sweep 0.6 — 1.0.
3. Edge case tests: separate small set of 100 items including embedded quotes,
   code blocks, list items, single-sentence responses, all-non-Korean lines.
4. **Baselines:**
   - **B1 — Random** uniform over 4 classes
   - **B2 — GPT-4 zero-shot**: prompt GPT-4 with "다음 문장의 문체를 골라주세요:
     합쇼체 / 해요체 / 문어체 / 해체"
   - **B3 — String-match (longest-suffix only)**: simple greedy match without
     priority-ordering, no quote stripping
5. **Inter-annotator agreement**: Cohen's κ on the 1,000-sentence Set B (3
   annotators × 200-sample overlap)

**Metrics:** macro-F1 across 4 buckets, per-bucket F1, response-level consistency
F1, κ, edge-case error rate.

**Outputs:** Table 4. Figure 3 (per-bucket confusion matrix as heatmap).

**Expected ranges:**
- Heuristic macro-F1: **0.88 – 0.94** (deterministic morphological matching is
  very strong for clear cases)
- Per-bucket F1: 합쇼체 **0.95+**, 해요체 **0.92+**, 문어체 **0.88+**, 해체 **0.75–0.85** (the
  weakest bucket — under-covered for safety, by design)
- GPT-4 zero-shot macro-F1: **0.80 – 0.88** (loses to the heuristic on clarity
  but better on ambiguous bare-다 cases)
- Inter-annotator κ: **≥ 0.85**

### E3 — Audience-following adherence rate
**Goal:** when `--audience 어린이` is requested, does the output actually look
child-targeted?

**Procedure:**
1. Generate Set C (2,400 outputs, 200 passages × 4 audiences × 3 registers)
2. For each output, compute:
   - **Mean word frequency** (using Korean modu-corpus frequency lookup)
   - **Sentence length p95** (chars)
   - **Sino-Korean fraction** (헌법 / 변호사 / 광합성 → Sino-Korean; vs. 떡 / 흙 / 별 → native)
3. **Auto-axis-classification**: train a simple logistic regression on the (4
   features → 4 audience classes) using the human-rated subset; report
   classifier accuracy
4. **Human ratings** (Set C): 3 native annotators rate each output's
   "audience_match" (1–5). ≥4 = adherent.
5. Compare `mixed` (no injection) vs each explicit audience setting → adherence
   rate per setting
6. **Confound check**: control for passage difficulty (technical Wikipedia vs.
   children's-book paragraph) — confirm adherence effect is not just passage-led

**Metrics:** mean rating per audience, adherence rate (≥4), classifier accuracy,
per-feature mean shift.

**Outputs:** Table 5. Figure 4 (4 boxplots showing classifier features distribute
by audience).

**Expected ranges:**
- 어린이 vs 전문가 mean-word-frequency separation: **+30% to +50%** (clear)
- 일반 vs 중고급중학교생 separation: **+5% to +15%** (murky — honest discussion in
  the paper)
- Human adherence rate (어린이): **60% – 78%**
- Human adherence rate (전문가): **65% – 80%**
- Human adherence rate (일반): **80% – 90%** (high — it's the default)
- Mixed (no injection) "default" audience: **80%+ rated as 일반** (proves the
  injection is doing something)

### E4 — End-to-end SFT comparison (the headline experiment)
**Goal:** do the filters move downstream model quality on Korean benchmarks?

**Setup (4 variants):**

| Var | SFT corpus |
|---|---|
| **V1** | 120k MT-translated Alpaca/OpenOrca/FLAN (NO filters) + 15k grounded (NO filters) |
| **V2** | V1 minus translationese-flagged (drop ~25–35% of translated → ~75–85k kept) + 15k grounded |
| **V3** | V1 minus register-mixed (drop ~15–25% of mixed responses → ~90–100k kept) + 15k grounded |
| **V4** | All filters + the **register × audience sweep** (3 registers × 4 audiences = 12 generations per passage = 60k grounded). MT side: V2's filtered subset |

**Training (identical hyperparameters across variants):**
- LR 1e-5, cosine schedule, 3% warmup
- 3 epochs, global batch 256 tokens (8M token batches), seq 4096
- bf16, full FT (no LoRA — the model size is fine for that on 16× H200)
- Parallelism: TP=2, PP=1, DP=8
- Approx. **~6 H200-hours per variant** = ~24 GPU-hours per seed × 3 seeds × 4 variants = **~288 GPU-hours total** (well within a 2-week budget on 16× H200)

**Evaluation:**
- KMMLU (45 subjects, 0-shot, accuracy)
- KoBEST (5 tasks, macro-F1)
- HAE-RAE Korean knowledge subsection (accuracy)
- CLIcK (cultural/linguistic, accuracy)
- MMLU (English regression — 40-subject subset, accuracy)
- GSM8K (English regression — full test, accuracy)

**Statistical test:** paired bootstrap (1000 resamples) over benchmark items;
report 95% CI on each delta.

**Outputs:** Table 6 (Korean benchmarks), Table 7 (English regression), Figure 5
(per-variant gains across benchmarks).

**Expected ranges (Δ from V1 baseline):**
- KMMLU (V4 − V1): **+1.5 to +3.0 points absolute**
- KoBEST (V4 − V1): **+0.8 to +2.0 points absolute**
- HAERAE (V4 − V1): **+1.0 to +2.5 points absolute**
- CLIcK (V4 − V1): **+1.5 to +3.5 points absolute**
- MMLU (V4 − V1): **−0.5 to +0.5 points** (must be ≥ −1.0; otherwise V4 is too
  aggressive and damaged English)
- GSM8K (V4 − V1): **−1.0 to +0.5 points** (similar bound)

V2 alone and V3 alone should each give ~50% of V4's gain — additive but not
fully so.

### E5 — Diversity-grid ablation (register × audience sweep)
**Goal:** does the 3×4 sweep produce more diverse training data than the
single-axis baseline?

**Procedure:**
1. Generate 10,000 outputs per condition:
   - **Cond-A**: `register=mixed, audience=mixed` (no axis control)
   - **Cond-B**: `register × audience sweep` (3 × 4 = 12 conditions, ~833 per cell)
2. Compute on each pool:
   - **Self-BLEU-4** (Zhu et al. 2018): lower = more diverse
   - **Type-token ratio** at token-bin = 10k (vocab richness)
   - **5-gram coverage**: % of distinct 5-grams in a held-out evaluation pool
     reachable by the training pool
   - **Embedding-based diversity**: mean cosine distance between random pairs
     using a multilingual embedding model (e.g., `intfloat/multilingual-e5-base`)
3. Cross-condition style separation: train a simple classifier (KoBERT) to
   predict register/audience from generated text; accuracy above chance ≡ the
   axes are realized in the data

**Outputs:** Table 8 (diversity metrics), Table 9 (axis-classifier accuracy).

**Expected:**
- Self-BLEU-4 (Cond-A): **0.30 – 0.40**
- Self-BLEU-4 (Cond-B sweep): **0.18 – 0.25** (lower = more diverse)
- 5-gram coverage (Cond-B / Cond-A ratio): **1.5x – 2.5x**
- Axis-classifier register accuracy: **0.75 – 0.90** (axes are realized)
- Axis-classifier audience accuracy: **0.55 – 0.70** (weaker — confirms E3's
  finding that audience is partially realized)

### E6 — Human pairwise preference evaluation
**Goal:** native Korean speakers prefer V4 outputs over V1 baseline outputs.

**Procedure:**
1. Generate Set D pairwise (V4 vs V1, V4 vs V2, V4 vs V3) — 600 pairs
2. Random pair-order, blind to model
3. 3 native annotators per pair; each rates on a 5-point preference scale
   (strongly prefer A / mild A / tie / mild B / strongly prefer B)
4. Compute **win rate** (A is preferred — count strongly + mild) and **adjusted
   win rate** (Bradley-Terry model on the 5-point ratings)
5. **Inter-annotator κ** at the 5-bucket level

**Metrics:** win rate, BT-adjusted score, κ, McNemar's test for paired
preference difference.

**Outputs:** Table 10. Figure 6 (win rate per filter variant, with 95% CI).

**Expected ranges:**
- V4 win rate over V1: **58% – 70%**
- V4 win rate over V2 (translationese alone): **52% – 60%** (modest — V2 is
  already filtered)
- V4 win rate over V3 (register alone): **53% – 62%**
- Inter-annotator κ: **≥ 0.55** (pairwise is harder than per-bucket)
- McNemar's p-value (V4 vs V1): **p < 0.001** expected

### E7 — Filter precision-recall calibration & failure-mode analysis
**Goal:** show how filter thresholds affect drop rate vs. precision; document
common failure modes for the discussion section.

**Procedure:**
1. Sweep `--min-signals` (translationese) and consistency threshold (register)
   over their ranges
2. Plot precision-recall curves
3. Sample 100 errors per filter: native-flagged-as-MT (false-positive) and
   MT-passed-as-native (false-negative)
4. Categorize errors and report rates for the paper's Section 8

**Outputs:** Figure 7 (PR curves), Table 11 (failure-mode breakdown).

**Expected dominant failure modes:**
- Translationese FP: short formal-Korean text triggers `it_will_be_ending`
  feature
- Translationese FN: fluent translator (human-translated) evades the
  heuristics (this is by design — heuristics target sloppy MT, not skilled HT)
- Register FP: literary text mixing 한다/이다 with 합니다 in dialogue tag → flagged
  as mixed; some annotators would say this is fine
- Register FN: very short responses (<3 sentences) pass without enough signal

---

## 3. Annotation logistics

See [annotation_guidelines.md](annotation_guidelines.md) for the rubric and
the calibration procedure. Summary:

1. **Recruit** 3–5 native Korean annotators (university students with linguistics
   coursework preferred). Hourly rate at or above Korean minimum wage (10,320
   KRW/hr 2026).
2. **Pilot batch** of 40 items per annotation set. Compute κ; if < 0.70, refine
   guidelines and re-pilot.
3. **Production batches** of 200 items at a time. 10% blind-gold (re-rated by
   the lead annotator) for ongoing calibration.
4. **Adjudication** for disagreements ≥ 2 buckets apart on the 5-point pairwise
   scale; an adjudicator (Korean-linguistics PhD or equivalent) breaks ties.
5. **Pay-per-batch with a quality bonus** (see Korean Playbook §4.4 rate card).

---

## 4. Statistical analysis

| Test | Used in | Notes |
|---|---|---|
| Paired bootstrap (1000 resamples) | E4, E5 | report 95% CI on Δaccuracy per benchmark |
| McNemar's test | E1, E2, E6 | for paired classification on the same items |
| Cohen's κ | E1, E2, E3, E6 | inter-annotator agreement |
| Bradley-Terry model | E6 | aggregate pairwise preferences |
| ANOVA + Tukey HSD | E3 | per-audience differences in feature distributions |

Report all p-values and CIs. Use scipy.stats.bootstrap and statsmodels.

---

## 5. Reproducibility checklist (paper appendix)

- [ ] Pinned `requirements.lock.txt`
- [ ] Frozen git commit hash (`8ff51d7`)
- [ ] Three seeds per condition; report mean ± std
- [ ] Hand-labeled sets released (with annotator IDs anonymized) under permissive
  license
- [ ] Filter calibration thresholds shipped in repo
- [ ] All generated SFT pools (V1–V4) released as JSONL
- [ ] Eval harness commit pinned (lm-eval-harness)
- [ ] Trained checkpoints released on HuggingFace Hub (or at least V4)

---

## 6. Timeline (single PI + 3 annotators + 1 engineer)

| Week | Milestone |
|---|---|
| 1 | Recruit annotators; pilot Set A and B (40 items each); compute κ |
| 2 | Annotation Set A complete (1,200 items); E1 detector calibration; E1 baselines |
| 3 | Annotation Set B complete (1,500 items); E2 detector evaluation; E2 baselines |
| 4 | Generate Set C and Set E; annotate; run E3 and E7 |
| 5 | Train V1 SFT (baseline). Eval V1 |
| 6 | Train V2 + V3 SFT. Eval both |
| 7 | Train V4 SFT (with sweep). Eval. Run E5 diversity metrics |
| 8 | Pairwise eval (Set D, 600 pairs). E6 analysis |
| 9 | Statistical analysis; figure production; first paper draft (use the template) |
| 10 | Internal review + revision |
| 11–12 | Submission preparation; supplement materials; release artifacts |

---

## 7. Cost estimate

| Item | Cost |
|---|---|
| GPU (288 H200-hours × $3/hr cloud rate) | ~$870 (or ~free on your owned 16× H200) |
| Korean annotators (600 person-hours × 25k KRW/hr) | ~$10,800 USD |
| GPT-4 baseline judging (E1, E2) | ~$200 |
| Open-source eval harness, KMMLU/HAERAE/CLIcK datasets | free |
| **Total** | **~$11,900** + 12 weeks |

This is achievable for a single research group.
