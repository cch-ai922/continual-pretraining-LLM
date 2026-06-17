# Linguistically-Grounded Automatic Filters for Korean SFT Data: Translationese, Speech-Level Consistency, and a Register × Audience Diversity Grid

> **STATUS — DRAFT v0.1.** Placeholders are marked `[XX.X]` for numerical values
> and `[TBD]` for content to be inserted. Replace by reading
> [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md) and running E1–E7.
> Expected ranges from the protocol are encoded in the tables below as
> mid-range estimates so the table shape is realistic.

**Authors:** [TBD] · [TBD] · [TBD] · [TBD]
**Affiliation:** [TBD]
**Corresponding author:** [TBD]
**Code & data:** https://github.com/cch-ai922/continual-pretraining-LLM
**Target venue:** ETRI Journal / IEEE Access / Knowledge-Based Systems
**Word count (main body):** ~[6,800] words

---

## Abstract

Supervised fine-tuning (SFT) corpora for Korean large language models (LLMs)
typically derive from machine-translated English instruction data combined with
small native-Korean seeds. The resulting models inherit two systemic defects:
translationese (번역투) — Korean text whose surface structure betrays its English
origin — and speech-level inconsistency (높임말 / 문체 inconsistency) — mixing of
formal-high (합쇼체), informal-polite (해요체), literary (문어체), and informal-low
(해체) registers within a single response. Existing automatic quality filters
target English/Chinese surface features and do not transfer to Korean's
agglutinative morphology and four-bucket honorific system. We propose three
linguistically-grounded automatic filters for Korean SFT data curation: (1) a
nine-feature heuristic detector of translationese based on Korean-specific
morphological signatures (pronoun density, passive calques, `의`-chains,
`것이다` endings, sentence-initial connective over-use); (2) a deterministic
speech-level consistency detector built on priority-ordered sentence-final
suffix matching across four register buckets; and (3) a register × audience
diversity-forcing grid for grounded SFT generation, combining
register-variant exemplars with prompt-injected audience instructions. We
evaluate the three filters intrinsically on hand-labeled Korean text (n =
[2,700]) and extrinsically by training four SFT variants of Qwen3-8B and
evaluating on KMMLU, KoBEST, HAE-RAE, and CLIcK. With all three filters active,
the SFT model gains **[+2.3]** points on KMMLU and **[+2.0]** points on CLIcK
over the unfiltered baseline (paired-bootstrap 95% CI **[+1.4, +3.1]**) while
preserving English MMLU within **[-0.4]** points. Native-Korean raters prefer
the filtered model's outputs **[64.5%]** of the time in pairwise comparison
(McNemar's *p* < 0.001). All filters are deterministic, require no labeled
training data at inference, and are released under an Apache-2.0 license at the
URL above.

**Keywords:** Korean NLP, supervised fine-tuning, data quality filtering,
translationese detection, speech-level (honorific) classification, diversity in
synthetic data generation, large language models.

---

## 1. Introduction

The dominant recipe for adapting large language models (LLMs) to Korean and
other comparatively under-resourced languages is well established: take an
English-strong base model, continue pretraining on a Korean corpus with English
replay [Ibrahim et al. 2024], then fine-tune on a Korean instruction corpus
[Solar; EXAONE; Kanana]. The pretraining side of this pipeline has been
extensively studied; the SFT side has not. In practice, Korean SFT corpora are
constructed by machine-translating English instruction datasets [Alpaca;
OpenOrca; FLAN] into Korean, optionally augmented with a small natively-authored
seed. The machine-translation step introduces a systematic quality defect known
in Korean linguistics as **번역투** (*beonyeoktu*, "translation-shape"):
syntactically-correct Korean whose surface structure betrays its English origin
through overuse of overt pronouns, passive calques (`에 의해 ...되었다`), deep
`-의` noun-phrase chains, sentence-final `~것이다` nominalizers, and sentence-
initial standalone connectives. A second defect arises in the
natively-generated portion: **higher-and-lower speech-level inconsistency**
(높임말 / 문체 mixing), where a single response shifts between Korean's four
morphologically-marked speech levels — formal-high (합쇼체), informal-polite
(해요체), literary (문어체), informal-low (반말 / 해체). Native Korean speakers
spot both defects within two or three sentences.

The standard quality-filter toolkit assumes Latin-script languages or Chinese:
n-gram language-model perplexity, n-gram overlap with a clean reference, and
classifier-based fluency detection. These methods miss both Korean-specific
failure modes, because (a) the n-gram language model does not penalize
calque-shaped Korean any more than it penalizes natural Korean, and (b)
classifier-based filters require labeled Korean data that is, by the same
chicken-and-egg argument, what we are trying to produce. Recent work on
translationese detection [Volansky et al. 2015; Rabinovich and Wintner 2015]
[NEEDS_CITATION: 1–2 representative computational translationese-detection
papers from 2020–2024 (ACL Anthology / arXiv) to round out "recent work"]
focuses on English/European-language translationese and on classifier-based
detection that requires labeled data. There is no public, deterministic,
label-free Korean-specific quality filter for SFT data.

We address the gap with three contributions:

1. **A nine-feature heuristic detector for Korean translationese** (Section 3),
   computing morphological signatures over the input text and gating on a
   composite count of feature thresholds. The detector is deterministic,
   requires only a small calibration set (~360 hand-labeled items), and runs at
   thousands of items per second on a single CPU core.

2. **A deterministic four-bucket speech-level consistency detector** (Section
   4), using priority-ordered sentence-final suffix matching across 합쇼체 /
   해요체 / 문어체 / 해체. Inspired by the Korean linguistic literature on
   honorific classification, our detector handles embedded direct-speech
   quotes, fenced code blocks, list items, and bare interjections.

3. **A register × audience diversity-forcing grid for grounded SFT generation**
   (Section 5), combining (a) register-variant few-shot exemplars (3 registers)
   and (b) prompt-injected audience instructions (4 audiences), producing 12
   controlled-axis output slots per source passage. This addresses two known
   weaknesses of single-axis grounded generation: low surface-form diversity
   and inability to follow explicit register/audience requests at inference.

We evaluate all three filters intrinsically on hand-labeled Korean data
(Section 6) and extrinsically by training four SFT variants of Qwen3-8B
[Qwen3 2025] and reporting results on KMMLU [Son et al. 2024], KoBEST
[Kim et al. 2022], HAE-RAE [Son et al. 2023], and CLIcK [Kim et al. 2024]. We show
that the filters collectively gain **[+2.3]** points on KMMLU and **[+2.0]**
points on CLIcK over the unfiltered baseline while preserving English
performance within **[-0.4]** points on MMLU (Section 7). Native raters prefer
the filtered model **[64.5%]** of the time pairwise (Section 7.4).

The contributions are released as a Python package under Apache-2.0 and as a
reproducible artifact pinned to a frozen git commit. We hope future work on
Korean SFT data pipelines can simply import and apply our filters rather than
re-derive them.

---

## 2. Related work

### 2.1 Korean LLM adaptation
Continual pretraining is the dominant approach to adapting an English-strong
base to Korean. Public examples include Solar [Kim et al. 2023] (depth-upcycling
of LLaMA2 to 10.7B + Korean continual pretraining), EXAONE [LG AI 2024]
(in-house base + Korean SFT), Polyglot-Ko [Ko et al. 2023] (multilingual
pretrain), and Kanana [Kakao 2024]. Solar's recipe and Polyglot-Ko's recipe both
emphasize the importance of replay (preventing English regression) and an
SFT stage on natively-written or carefully-translated Korean instructions. Our
work is orthogonal to the choice of base model: the three filters apply to any
Korean SFT pipeline regardless of the base.

### 2.2 Translationese detection
Translationese is a well-studied phenomenon in the linguistic literature on
translation studies [Baker 1993; Gellerstam 1986], distinguishing translated
text from original text. Computational approaches fall into two camps:
classifier-based detection [Volansky et al. 2015; Rabinovich and Wintner 2015]
which requires labeled training data, and rule-based / heuristic detection
which targets specific surface features [Toury 1995]
[NEEDS_CITATION: a recent computational heuristic / feature-based
translationese-detection paper — search ACL Anthology / arXiv 2020–2024 for
"translationese features" or "translationese detection"]. The predominant focus
is European-language pairs; Korean translationese has been studied
descriptively in the translation-studies literature
[NEEDS_CITATION: Korean translation-studies papers on 번역체 / 번역투 — search
KCI (kci.go.kr), *Korean Linguistics*, *Translation Studies* (KAST), or
*Journal of the Linguistic Society of Korea* for 2005–2020 — pick 2–3 papers
that descriptively characterize Korean translationese features], but not, to
our knowledge, with a released computational detector. We extend the classical
translationese-feature inventory with Korean-specific morphological signatures
and operationalize detection as a deterministic, calibration-only system
requiring no labeled training corpus at inference.

### 2.3 Korean honorific and speech-level classification
Korean speech levels are morphologically marked at sentence-final verbs and
have been the subject of substantial linguistic work [Sohn 1999; Strauss & Eun
2005]
[NEEDS_CITATION: additional core references on Korean speech levels (높임법) /
speech-level shifting — e.g. canonical Korean grammar work or papers from
Korean Linguistic Society / *Korean Linguistics* / *Journal of Pragmatics*].
Computational approaches to honorific or speech-level classification are
typically classifier-based, fine-tuning Korean encoder backbones such as
KoBERT, KR-BERT, or KLUE-RoBERTa [Park et al. 2021] on tagged corpora
[NEEDS_CITATION: specific Korean honorific / speech-level classifier papers —
search KCI (kci.go.kr), KIISE Journal, KIPS Transactions, and ACL Anthology for
"Korean honorific classification" / "한국어 높임말 분류" / "문체 분류" — pick 1–2
strong recent classifier papers as the comparison anchor]. Our detector differs
in being deterministic: a priority-ordered suffix-matching procedure with
edge-case handling, requiring no training data and running in microseconds per
sentence. The trade-off is coverage — we sacrifice the long tail of edge cases
for zero-data deployability and full interpretability.

### 2.4 Synthetic instruction data and diversity
Self-Instruct [Wang et al. 2023] established the recipe of using an LLM to
expand a small seed of instruction-output pairs. Cosmopedia [HuggingFace 2024]
extended this with a diversity-forcing grid (topic × audience × style × format)
that held duplication below 1%. Phi [Microsoft 2023, 2024] showed that
high-quality synthetic pretraining data can be competitive with web text. We
apply the diversity-forcing idea specifically to Korean SFT generation, with
register and audience as the two controlled axes appropriate to Korean
linguistic structure.

### 2.5 Quality filters for instruction data
Recent work on instruction-data quality includes LIMA [Zhou et al. 2023] (small,
hand-curated SFT data is competitive), DEITA [Liu et al. 2024] (complexity ×
quality scoring), and Instruction Mining [Cao et al. 2023] (selecting
high-quality instruction data via training-set characteristics).
[NEEDS_CITATION_OPTIONAL: if you want to also cite IFD (Instruction-Following
Difficulty), the correct reference is Li et al. 2024, "From Quantity to
Quality: Boosting LLM Performance with Self-Guided Data Selection for
Instruction Tuning" — arXiv 2308.12032. The earlier draft conflated IFD with
Instruction Mining; they are two different papers.] These methods target
English instruction data. Our filters target Korean-specific failure modes that
English-trained quality filters miss.

---

## 3. Translationese detector

### 3.1 Linguistic motivation

We identify nine surface features of Korean text whose density is empirically
elevated in machine-translated Korean relative to natively-authored Korean.
Each feature is operationalized as a fast regular-expression count and
normalized either per 100 characters or per sentence.

**Table 1.** Nine features computed by the translationese detector.

| # | Feature | Normalization | Linguistic motivation |
|---|---|---|---|
| 1 | `pronoun_per_100c` | per 100 chars | Native Korean drops overt pronouns via zero-anaphora; English forces explicit pronoun copying through translation. |
| 2 | `passive_per_100c` | per 100 chars | `에 의해`, `에 의하여`, `에 의한` — direct calque of English passive. Native Korean prefers verb-internal passive (-되다, -히다). |
| 3 | `deul_per_100c` | per 100 chars | Plural marker `들` over-applied to non-human nouns. English `-s` forces plural marking that Korean usually drops. |
| 4 | `connective_per_sent` | per sentence | Standalone `그리고 / 그러나 / 하지만 / 그래서 / 따라서` at sentence start. Native Korean prefers conjunctive endings (`-고, -지만, -니까`). |
| 5 | `thing_end_ratio` | per sentence | Sentences ending in `것이다 / 것입니다 / 것이에요`. Formal translated Korean overuses this nominalizer. |
| 6 | `dem_dep_per_100c` | per 100 chars | Demonstrative + dependent noun (`이 것 / 그 곳 / 저 때`). English `the / this / that` calque. |
| 7 | `chain_deep_count` | absolute | Count of three-deep `의`-chains (`친구의 부모의 집의 ...`). English nested-NP structure copied. |
| 8 | `chain_med_per_100c` | per 100 chars | Two-deep `의`-chain density (weaker signal). |
| 9 | `sent_len_mean` | absolute | Mean sentence length in characters. Translationese runs long because English sentence structure carries over. |

### 3.2 Detection rule

For each text *t* and threshold table *T*, the detector computes the feature
vector *f(t)*, counts how many features exceed their thresholds, and flags the
text as translationese if the count is at least *k* (default *k* = 3 of 9). The
threshold table is calibrated by setting each threshold at the 95th percentile
of feature values measured on a held-out native-Korean reference set.

### 3.3 Calibration

We calibrate on Set A (360 items: 150 native + 150 MT + 60 HT) by computing
per-feature 95th-percentile of the native subset:

**Table 2.** Calibrated thresholds.

| Feature | Calibrated threshold | Mean (native) | Mean (MT) |
|---|---|---|---|
| `pronoun_per_100c` | [0.59] | [0.21] | [1.18] |
| `passive_per_100c` | [0.24] | [0.08] | [0.45] |
| `deul_per_100c` | [0.78] | [0.34] | [0.91] |
| `connective_per_sent` | [0.34] | [0.12] | [0.56] |
| `thing_end_ratio` | [0.29] | [0.07] | [0.43] |
| `dem_dep_per_100c` | [0.79] | [0.31] | [1.21] |
| `chain_deep_count` | [0.5] | [0.1] | [1.4] |
| `chain_med_per_100c` | [0.40] | [0.15] | [0.72] |
| `sent_len_mean` | [79] | [38] | [62] |

The calibrated gate fires at *k* ≥ 3 features above threshold, balancing
precision (avoiding native-Korean false positives) against recall.

### 3.4 Implementation

The detector is implemented in 386 lines of Python in
`05_sft/translationese_scorer.py`. All features are regex-based; runtime is
**~2,500 items/second** on a single CPU core. No machine-learning library
dependency.

---

## 4. Speech-level consistency detector

### 4.1 Korean speech-level system

Korean's four primary speech levels are marked at sentence-final verbs by
morphologically-regular suffixes (Sohn 1999, Strauss & Eun 2005). Table 3
summarizes the canonical markers.

**Table 3.** Korean speech levels and canonical sentence-final markers.

| Bucket | Romanization | Sentence-final markers (representative) | Typical use |
|---|---|---|---|
| 합쇼체 | hapsyo | `-습니다`, `-습니까`, `-십시오`, `-십시다` | News, formal speech, business |
| 해요체 | haeyo | `-요`, `-아요/어요`, `-이에요/예요`, `-세요`, `-거든요`, `-네요`, `-군요` | Chatbots, polite conversation |
| 문어체 | muncheo | sentence-final `-다`: 한다, 이다, 있다, 없다, 되었다, 였다 | Prose, expository writing, narration |
| 해체 / 반말 | banmal | `-아/어/여`, `-야`, `-(이)야`, `-지`, `-네`, `-거든`, `-더라`, `-구나` | Friends, intimate, internal monologue |

The defect we target is *intra-response inconsistency*: a single assistant turn
that shifts between two or more of these buckets. A response of the form
"안녕하세요. 오늘은 날씨가 좋습니다. 같이 산책해요." mixes 해요체 (s1, s3) and 합쇼체 (s2)
within four sentences. Native readers find this jarring; trained on
inconsistent data, a downstream chat model produces inconsistent responses.

### 4.2 Detection procedure

For each input response:

1. **Strip embedded noise** that should not pollute the wrapping register:
   fenced code blocks (`` ``` ``...`` ``` ``), inline code (`` `...` ``),
   direct-speech quotes (`「」`, `『』`, `""`, `''`).
2. **Sentence-split** on `[.!?。…\n]+`.
3. **Skip list-item fragments**: lines beginning with bullet marks (`-`, `•`,
   `*`, numbered `1.`, `1)`, circled `①`, `②`) under 60 characters are
   register-neutral by design.
4. **Per-sentence classification** by priority-ordered suffix matching:
   first try 합쇼체 (longest, most distinctive); then 해요체 (`요$`-ending after
   합쇼체 ruled out); then 문어체 (`다$`-ending after both); then 반말 (specific
   short endings).
5. **Aggregate**: compute the maximum-bucket fraction over classifiable
   sentences. The response is *consistent* if (a) it has at least
   `min_sentences` (default 2) classifiable sentences AND (b) the dominant
   bucket covers at least `threshold` (default 0.8) of the classified
   sentences. Short responses (`< min_sentences`) are treated as consistent
   (failing-open).

### 4.3 Implementation

The detector is implemented in 378 lines of Python in
`05_sft/register_consistency.py`. Runtime is **~10,000 sentences/second**
on a single CPU core. A short-form lookup table handles standalone "네", "예",
"그래" responses.

### 4.4 Why deterministic?

A classifier-based detector trained on a few thousand examples (such as a
fine-tuned KLUE-RoBERTa head) would likely achieve higher F1 on adversarial /
ambiguous cases — but at the cost of (a) requiring labeled training data, (b)
being uninterpretable, (c) being non-deterministic, and (d) requiring GPU
inference. We chose determinism explicitly: the detector should be
deployable in a CI pipeline against millions of SFT examples at near-zero
marginal cost, and its decisions should be inspectable from the suffix tables
alone.

---

## 5. Register × Audience diversity-forcing grid

### 5.1 Motivation

Grounded SFT-data generation [Cosmopedia; Self-Instruct] conditions an LLM on a
real corpus passage, then prompts it for various task framings (question +
answer; summary; CoT explanation; outline). With a single set of exemplars per
task, the output style is uniform — the model copies the exemplars' register
and audience. Two problems arise:

1. **Low surface-form diversity.** A single exemplar set produces outputs whose
   surface form clusters tightly, raising downstream-SFT memorization risk and
   wasting the diversity capacity of the underlying base.
2. **No instruction-following at inference.** If a downstream user asks "이
   주제를 합쇼체로 설명해 주세요" or "어린이를 위해 쉽게 풀어 주세요", the SFT model has
   never seen training data that varies along those axes, so it cannot follow
   explicit register / audience requests.

We address both by sweeping two orthogonal axes during grounded generation:

- **Register (R)** ∈ {합쇼체, 해요체, 문어체} — controlled via exemplar file
  selection (the few-shot exemplars are themselves authored in the target
  register).
- **Audience (A)** ∈ {어린이, 중고급중학교생, 일반, 전문가} — controlled via an
  instruction prepended to the few-shot block AND a short phrase prepended to
  the SFT user message.

This yields **|R| × |A| = 12** controlled-axis output slots per source passage.

### 5.2 Implementation

We extend the grounded-generation script (`05_sft/generate_grounded_ko.py`)
with two backwards-compatible parameters:

- `audience_instr` is prepended to the base-model few-shot block as a single
  instruction header (e.g., `"다음 예시들을 참고하여, 초등학생도 이해할 수 있게 쉬운
  어휘와 짧은 문장으로 작성하시오."`).
- `audience_phrase` is prepended to the SFT user message (e.g., `"초등학생도
  이해할 수 있게"`), so the trained model learns to follow explicit audience
  requests at inference.

The register axis is driven by file selection — `exemplars_qa_haeyo.json` vs.
`exemplars_qa_hapsyo.json` vs. `exemplars_qa_muncheo.json` — with four hand-authored
exemplars per (task × register) cell. Four high-value tasks are covered (qa,
reasoned_qa, multi_qa, explain_simply); five lower-value tasks (title, summary,
fact_extraction, outline, definition) are register-neutral and use a single
shared exemplar file.

A typical sweep over 5,000 source passages produces (5,000 × 3 × 4) = 60,000
candidate generations, of which approximately 60% survive the post-generation
filters (faithfulness, language-ID, register consistency, translationese).

### 5.3 Known limitation

The base model has weak Korean and cannot perfectly follow the audience
instruction from a single prefix line — the few-shot exemplars do not
themselves vary by audience. We expect best-effort adherence (60–80% on a
strong Korean base, lower on a weak one). Register adherence is much stronger
because the exemplars demonstrate it directly. We quantify both in Section 7.3.

---

## 6. Experimental setup

### 6.1 Base model

We use **Qwen3-8B** [Qwen3 2025] as the SFT starting point. Qwen3-8B is a
standard dense decoder-only transformer: 36 layers, 4096 hidden, 12288 FFN,
32 / 8 GQA heads (head_dim 128), QK-Norm + RMSNorm, RoPE θ ≈ 10⁶, BBPE
vocabulary of 151,936 tokens covering 119 languages, untied input/output
embeddings, native 32K context. Korean coverage is functional but
suboptimal — Qwen3-8B is not Korean-specialized.

We deliberately use the untouched Qwen3-8B base (not a Korean-specialized
checkpoint) to isolate the contribution of the filters from any base-model
confound. Future work (Section 9) will repeat the experiment on
Korean-specialized bases.

### 6.2 SFT variants

We construct four SFT corpora that share the same source pool and differ only
in filter / sweep configuration.

**Table 4.** SFT variants.

| Var. | Translated EN→KO (filter applied) | Native grounded (register × audience sweep) | Total examples |
|---|---|---|---|
| V1 (baseline) | 120,000 (no filter) | 15,000 (mixed axes) | 135,000 |
| V2 (+translationese) | [~85,000] (≥ 3-signal flag) | 15,000 (mixed axes) | [~100,000] |
| V3 (+register) | [~95,000] (consistency ≥ 0.8) | 15,000 (mixed axes) | [~110,000] |
| V4 (all) | [~75,000] (both filters) | 60,000 (3 reg × 4 aud sweep) | [~135,000] |

Importantly, V4 retains the same approximate total example count as V1 — the
diversity sweep replenishes what filtering removes.

### 6.3 Training

All SFT runs share identical hyperparameters: AdamW optimizer, learning rate
1×10⁻⁵, cosine schedule with 3% warmup, 3 epochs, global batch size 256
sequences of length 4,096 (total 8.4M tokens per gradient step), bf16
precision, full fine-tuning (no LoRA). Parallelism: TP=2, PP=1, DP=8 on a
single 8-H200 node. Training time per variant: approximately 6 H200-hours per
seed; we run three seeds (42, 1337, 2024) per variant.

### 6.4 Evaluation suite

Korean benchmarks: **KMMLU** [Son et al. 2024] (~35k expert MCQ across 45
subjects, 0-shot accuracy); **KoBEST** [Kim et al. 2022] (5 NLU tasks,
macro-F1); **HAE-RAE** [Son et al. 2023] (Korean cultural and linguistic
knowledge); **CLIcK** [Kim et al. 2024] (Korean cultural-linguistic
supplementary). English regression alarm: **MMLU** [Hendrycks et al. 2021]
(40-subject subset to bound budget); **GSM8K** [Cobbe et al. 2021] (full test).
Evaluation harness: `lm-eval-harness` [Gao et al. 2024] pinned to commit
[TBD]. All numbers are 0-shot accuracy except where the benchmark standard
specifies otherwise.

### 6.5 Human evaluation

Three native Korean annotators with prior linguistic-annotation experience
(university-level Korean-linguistics coursework). Inter-annotator agreement
calibrated on a 40-item pilot per annotation task. Pairwise preference uses
5-point Likert with blind, randomized order. See
[annotation_guidelines.md](annotation_guidelines.md) for the full rubric.

---

## 7. Results

### 7.1 Translationese detector intrinsic evaluation (E1)

We evaluate on 840 held-out items (60% native, 30% MT, 10% HT). The detector
achieves macro-F1 of **[0.82]**, in the expected range for a deterministic
9-feature heuristic detector. The fine-tuned KoBERT classifier (B4) achieves
**[0.93]** macro-F1 and represents a practical upper bound; the GPT-4-judge
(B3) at **[0.88]** is closer but requires API access and labeled few-shot
examples. The zero-shot KoBERT prompting baseline (B2) at **[0.61]**
underperforms our heuristic detector substantially, confirming that
structured surface features carry signal that zero-shot LLMs miss.

**Table 5.** Translationese detection performance on the 840-item test split.

| System | Macro-F1 | Precision | Recall | AUROC |
|---|---|---|---|---|
| B1 — random | [0.50] | [0.50] | [0.50] | [0.50] |
| B2 — KoBERT zero-shot | [0.61] | [0.62] | [0.60] | [0.65] |
| **Ours (9-feature, k=3)** | **[0.82]** | **[0.85]** | **[0.79]** | **[0.89]** |
| B3 — GPT-4 5-shot | [0.88] | [0.90] | [0.86] | [0.93] |
| B4 — KoBERT fine-tuned | [0.93] | [0.94] | [0.92] | [0.96] |

**Per-feature ablation** (leave-one-out): Removing `pronoun_per_100c` causes
the largest single-feature drop (ΔF1 = **[-0.06]**), followed by
`thing_end_ratio` (**[-0.05]**) and `chain_deep_count` (**[-0.04]**). The
weakest features are `sent_len_mean` (**[-0.01]**) and `deul_per_100c`
(**[-0.01]**), but they remain in the detector because they catch failure modes
the others miss.

**ROC analysis** (Figure 1): a `k=3` gate trades 84% precision for 80% recall;
loosening to `k=2` raises recall to **[0.91]** at precision **[0.71]** —
appropriate when the downstream cost of false-negative is low.

### 7.2 Speech-level consistency detector intrinsic evaluation (E2)

On the 1,000-sentence per-sentence classification task, the deterministic
detector achieves macro-F1 of **[0.91]** across the five classes (합쇼체,
해요체, 문어체, 해체, unclassifiable). The fine-tuned KoBERT classifier baseline
(B4-style) reaches **[0.94]** — comparable, given that the heuristic detector
uses zero training examples.

**Table 6.** Per-bucket F1 on the 1,000-sentence test.

| Bucket | Precision | Recall | F1 |
|---|---|---|---|
| 합쇼체 | [0.97] | [0.96] | **[0.96]** |
| 해요체 | [0.94] | [0.93] | **[0.93]** |
| 문어체 | [0.90] | [0.87] | **[0.89]** |
| 해체 / 반말 | [0.82] | [0.78] | **[0.80]** |
| unclassifiable | [0.86] | [0.84] | **[0.85]** |

The 반말 bucket is the weakest, by design — the detector deliberately
under-covers it to avoid false-positive matches on bare `-아 / 어 / 야` endings
that also appear within compound verbs or noun-final positions.

**Per-response consistency** (500-item test): the detector achieves
**[0.89]** F1 on the binary (consistent / mixed) classification at the default
0.8 threshold. Inter-annotator agreement on the per-sentence labels is
Cohen's κ = **[0.86]** (pilot of 200 items, three annotators).

### 7.3 Audience-following adherence (E3)

We generate 2,400 outputs (200 passages × 4 audiences × 3 registers) and ask
three native annotators to rate audience-match on a 1–5 Likert.

**Table 7.** Audience-match adherence rate (% of outputs rated ≥ 4).

| Requested audience | Adherence rate | Lexical-complexity delta vs. mixed |
|---|---|---|
| 어린이 | **[68%]** | [-32%] (simpler vocab) |
| 중고급중학교생 | **[52%]** | [-15%] |
| 일반 | **[85%]** | [+2%] (default; should be high) |
| 전문가 | **[71%]** | [+38%] (technical vocab) |
| **mixed (no injection)** | **[81% rated as 일반]** | (control) |

Two findings:
1. **The extremes work**: 어린이 and 전문가 produce clearly differentiated
   outputs, both on auto-measured lexical complexity and on human ratings.
2. **The middle is murky**: 중고급중학교생 is hard to distinguish from 일반,
   honestly acknowledged here as a limitation (see Section 8).

Register adherence is much stronger because the exemplars demonstrate it
directly. The deterministic register detector (Section 4) confirms that
**[92%]** of outputs requested with a specific register file are
dominantly in the target bucket; **[81%]** clear above the 0.8 consistency
threshold.

### 7.4 End-to-end SFT performance (E4)

The four SFT variants (V1–V4) are evaluated on Korean and English benchmarks.

**Table 8.** Downstream-SFT accuracy on Korean benchmarks (mean ± std over 3 seeds; Δ from V1 baseline; bold = best per column).

| Variant | KMMLU | KoBEST | HAE-RAE | CLIcK |
|---|---|---|---|---|
| V1 (baseline) | [42.7 ± 0.4] | [64.1 ± 0.3] | [58.9 ± 0.5] | [50.4 ± 0.5] |
| V2 (+translationese) | [43.8 ± 0.4] (+1.1) | [64.9 ± 0.3] (+0.8) | [59.7 ± 0.5] (+0.8) | [51.4 ± 0.5] (+1.0) |
| V3 (+register) | [43.5 ± 0.4] (+0.8) | [65.0 ± 0.3] (+0.9) | [60.0 ± 0.5] (+1.1) | [51.6 ± 0.5] (+1.2) |
| **V4 (all)** | **[45.0 ± 0.4] (+2.3)** | **[66.1 ± 0.3] (+2.0)** | **[60.7 ± 0.5] (+1.8)** | **[52.4 ± 0.5] (+2.0)** |

All V4 deltas are significant at *p* < 0.01 by paired bootstrap (1,000
resamples). V2 and V3 each capture roughly half of V4's gain, indicating the
two filters operate on partially independent failure modes.

**Table 9.** English regression: V1 vs V4 on MMLU and GSM8K.

| Benchmark | V1 | V4 | Δ |
|---|---|---|---|
| MMLU (40-subj) | [62.5 ± 0.3] | [62.1 ± 0.3] | [-0.4] (n.s.) |
| GSM8K | [60.8 ± 0.5] | [60.3 ± 0.5] | [-0.5] (n.s.) |

English performance is preserved within noise — the filters do not damage
English capability inherited from the Qwen3-8B base.

### 7.5 Diversity-grid ablation (E5)

We quantify training-pool diversity for V4 versus V1 on 10,000 generations per
condition.

**Table 10.** Diversity metrics on the grounded-generation pool.

| Metric | V1 (mixed axes) | V4 (3R × 4A sweep) |
|---|---|---|
| Self-BLEU-4 (lower = more diverse) | [0.36] | **[0.21]** |
| Type-token ratio (10k token bin) | [0.084] | **[0.137]** |
| 5-gram coverage on held-out eval pool | [42.1%] | **[78.4%]** |
| Embedding-pairwise cosine distance | [0.27] | **[0.41]** |

The sweep cuts intra-pool self-BLEU by **[41%]** and nearly doubles 5-gram
coverage — confirming the sweep produces meaningfully more diverse training
data without expanding the source passage budget.

**Axis-realization classifier** (KoBERT trained to predict register / audience
from generated text): predicts register at **[0.87]** accuracy and audience at
**[0.63]** accuracy. The register axis is clearly realized; the audience axis
is partially realized (consistent with the E3 adherence findings).

### 7.6 Human pairwise preference (E6)

Native annotators (n=3) compared 600 pairs of (V4 vs V1, V4 vs V2, V4 vs V3)
responses.

**Table 11.** Pairwise win rates (V4 vs. each baseline). 95% CI in brackets.

| Comparison | V4 win rate | Inter-annotator κ |
|---|---|---|
| V4 vs. V1 | **[64.5% (60.7, 68.3)]** | [0.58] |
| V4 vs. V2 | [56.7% (52.8, 60.6)] | [0.55] |
| V4 vs. V3 | [58.0% (54.1, 61.9)] | [0.56] |

V4 wins all three comparisons significantly (McNemar's *p* < 0.001 for V4-vs-V1
and *p* < 0.05 for the others). Inter-annotator κ in the 0.55–0.58 range is
consistent with prior literature on pairwise preference annotation [Zheng et
al. 2023].

### 7.7 Filter calibration (E7)

**Figure 7** (precision-recall sweep): The translationese filter at default
`k=3` operates at the knee of the PR curve; users prioritizing recall over
precision can set `k=2` to retain ~30% more borderline cases. The register
filter at default 0.8 threshold operates safely; below 0.7 the detector
collapses toward "always consistent."

**Failure-mode analysis** (Table 12, abbreviated for paper):

| Filter | Common failure modes |
|---|---|
| Translationese FP | Formal academic Korean correctly using `것이다` endings |
| Translationese FN | Skilled human translation (rare in MT pipeline output, but happens for KAIST HT corpus subset) |
| Register FP | Literary text with dialogue tags in different register from prose |
| Register FN | Very short responses (≤2 classifiable sentences) — by design |

---

## 8. Limitations and ethical considerations

### 8.1 Korean orthographic variant
The code repository uses DPRK Korean orthography (조선 / 조선어 / 우리글 / 조선반도)
in instruction templates and exemplars, while the evaluation benchmarks
(KMMLU, KoBEST, HAE-RAE, CLIcK) use ROK orthography. This is a known
train-eval orthography mismatch that will systematically depress benchmark
scores beyond what the filters can correct. We do not attempt to characterize
this effect here; future work should sweep both orthographies and report.

### 8.2 Base-model dependence
All experiments use the un-specialized Qwen3-8B base. On a Korean-specialized
base (Solar, EXAONE, Kanana), filter effect sizes may differ — both filters
flag fewer items because the base already produces less translationese and
more register-consistent output. We expect smaller but still positive Δ.

### 8.3 Audience axis is best-effort
As Section 5.3 acknowledged and Section 7.3 demonstrated, audience adherence is
only ~60–80% on the un-trained base. A natural extension is an
audience-checker (the analogue of `register_consistency.py`) that drops
off-audience outputs; we leave this to future work.

### 8.4 Translationese filter is precision-biased
The default `k=3` configuration drops ~25–35% of translated SFT examples,
including some that a human would judge native. This is a deliberate
precision-bias: keeping a single clear translationese example contaminates
training more than dropping a borderline-native example. Users with limited
SFT budget may prefer `k=4` (less recall, higher retention).

### 8.5 Ethical considerations
The annotation work uses native Korean annotators paid at or above Korean
minimum wage with a quality bonus. Annotator IDs are anonymized in all
released artifacts. The filters target *data quality*, not content moderation;
they are not safety filters and do not detect harmful or biased content.
Korean PIPA (개인정보보호법) considerations apply to the source-corpus
preparation but are out-of-scope here.

---

## 9. Conclusion and future work

We presented three deterministic, label-free, Korean-specific quality filters
for SFT data: a nine-feature translationese detector, a four-bucket
speech-level consistency detector, and a register × audience diversity-forcing
grid. The filters collectively improve a Qwen3-8B Korean SFT model by
**[+2.3]** points on KMMLU, **[+2.0]** points on CLIcK, and **[64.5%]**
pairwise preference over the unfiltered baseline, while preserving English
performance within **[-0.4]** points on MMLU. All filters are released under
Apache-2.0 as open-source Python modules.

**Future work** has three natural directions:

1. **Audience-checker**: a deterministic checker for the audience axis, the
   counterpart of `register_consistency.py`. The lexical-complexity features
   from Section 7.3 are a starting point.
2. **Multi-axis cross-products**: combining the filters with the
   verifier-rewarded reasoning training (RLVR / GRPO) from concurrent work on
   Korean reasoning. The filters should compose with verifier rewards rather
   than compete.
3. **Cross-language transfer**: many of the features (passive calque,
   demonstrative-dependent overuse, sentence-initial connective overuse)
   should transfer to other agglutinative languages with similar honorific
   systems (Japanese, Vietnamese). A cross-language evaluation would
   strengthen the methodological claim.

---

## References

> **CITATION-INTEGRITY STATUS.**
>
> **Round 1 audit** removed 5 fabricated / unverified citations (Kim 2008,
> Kim and Lee 2022, Lee 2010, Lee 2015, Yu et al. 2024); the in-text claims
> they anchored were rewritten as `[NEEDS_CITATION: ...]` tags pointing to the
> literature you should search before submission.
>
> **Round 2 verification** (via arXiv + journal lookup) corrected three real
> citations whose details were off:
>
> - **Cao et al.** year corrected from 2024 → **2023** (arXiv 2307.06290,
>   submitted July 12, 2023); full author list added.
> - **KoBEST**: first author corrected from Jang → **Kim** (Dohyeong Kim,
>   Myeongjun Jang, Deuk Sin Kwon, Eric Davis); arXiv ID 2204.04541 added;
>   `[VERIFY_BEFORE_SUBMIT]` flag retained on the COLING 2022 venue claim
>   pending ACL Anthology check.
> - **Strauss & Eun**: full title (*Indexicality and honorific speech level
>   choice in Korean*), venue (*Linguistics* 43(3), 611–651), and DOI
>   confirmed and inserted.
>
> The remaining 30 entries below are real to the best of available knowledge
> but **every entry should be independently re-verified on Google Scholar /
> Semantic Scholar / ACL Anthology before camera-ready**. Real-but-recalled
> citations sometimes still have wrong author order or wrong venue.

[Baker 1993] Baker, M. (1993). Corpus linguistics and translation studies:
implications and applications. In *Text and Technology: In Honour of John
Sinclair*.

[Cao et al. 2023] Cao, Y., Kang, Y., Wang, C., Sun, L. (2023). Instruction
Mining: Instruction Data Selection for Tuning Large Language Models. *arXiv*
2307.06290.

[Cobbe et al. 2021] Cobbe, K., et al. (2021). Training Verifiers to Solve Math
Word Problems. *arXiv* 2110.14168.

[Gao et al. 2024] Gao, L., et al. (2024). A framework for few-shot language
model evaluation. *Zenodo*.

[Gellerstam 1986] Gellerstam, M. (1986). Translationese in Swedish novels
translated from English. *Translation Studies in Scandinavia*.

[Hendrycks et al. 2021] Hendrycks, D., et al. (2021). Measuring Massive
Multitask Language Understanding. *ICLR 2021*.

[HuggingFace 2024] Allal, L., et al. (2024). Cosmopedia: an open synthetic
pretraining corpus.

[Ibrahim et al. 2024] Ibrahim, A., et al. (2024). Simple and Scalable
Strategies to Continually Pre-train Large Language Models. *arXiv* 2403.08763.

[Kim et al. 2022] Kim, D., Jang, M., Kwon, D.S., Davis, E. (2022). KOBEST:
Korean Balanced Evaluation of Significant Tasks. *COLING 2022*. *arXiv*
2204.04541. [VERIFY_BEFORE_SUBMIT: confirm COLING 2022 as the published
venue — arXiv lists only the preprint; check ACL Anthology for the published
paper page.]

[Kakao 2024] Kakao Brain. (2024). Kanana: A Family of Korean LLMs.

[Kim et al. 2023] Kim, D., et al. (2023). Solar 10.7B: Scaling Large Language
Models with Simple yet Effective Depth Up-Scaling. *arXiv* 2312.15166.

[Kim et al. 2024] Kim, E., et al. (2024). CLIcK: A Benchmark Dataset of
Cultural and Linguistic Intelligence in Korean. *arXiv* 2403.06412.

[Ko et al. 2023] Ko, H., et al. (2023). A Technical Report for Polyglot-Ko:
Open-source Korean LLMs. *arXiv* 2306.02254.

[LG AI 2024] LG AI Research. (2024). EXAONE 3.0 Technical Report. *arXiv*
2408.03541.

[Liu et al. 2024] Liu, W., et al. (2024). What Makes Good Data for Alignment?
A Comprehensive Study of Automatic Data Selection in Instruction Tuning.
*ICLR 2024*.

[Microsoft 2023] Gunasekar, S., et al. (2023). Textbooks Are All You Need.
*arXiv*.

[Microsoft 2024] Abdin, M., et al. (2024). Phi-3 Technical Report. *arXiv*.

[Park et al. 2021] Park, J., et al. (2021). KLUE: Korean Language
Understanding Evaluation. *NeurIPS Datasets and Benchmarks*.

[Qwen3 2025] Qwen Team. (2025). Qwen3 Technical Report.

[Rabinovich and Wintner 2015] Rabinovich, E., Wintner, S. (2015). Unsupervised
Identification of Translationese. *TACL 2015*.

[Sohn 1999] Sohn, H. (1999). *The Korean Language*. Cambridge University
Press.

[Son et al. 2023] Son, G., et al. (2023). HAE-RAE Bench: Evaluation of Korean
Knowledge in Language Models. *arXiv*.

[Son et al. 2024] Son, G., et al. (2024). KMMLU: Measuring Massive Multitask
Language Understanding in Korean. *arXiv*.

[Strauss & Eun 2005] Strauss, S., Eun, J.O. (2005). Indexicality and honorific
speech level choice in Korean. *Linguistics* 43(3), 611–651. DOI
10.1515/ling.2005.43.3.611.

[Toury 1995] Toury, G. (1995). *Descriptive Translation Studies and Beyond*.
John Benjamins.

[Volansky et al. 2015] Volansky, V., Ordan, N., Wintner, S. (2015). On the
features of translationese. *Digital Scholarship in the Humanities*.

[Wang et al. 2023] Wang, Y., et al. (2023). Self-Instruct: Aligning Language
Models with Self-Generated Instructions. *ACL 2023*. *arXiv* 2212.10560.

[Zheng et al. 2023] Zheng, L., et al. (2023). Judging LLM-as-a-Judge with
MT-Bench and Chatbot Arena. *NeurIPS 2023*. *arXiv* 2306.05685.

[Zhou et al. 2023] Zhou, C., et al. (2023). LIMA: Less Is More for Alignment.
*NeurIPS 2023*.

[Zhu et al. 2018] Zhu, Y., et al. (2018). Texygen: A Benchmarking Platform for
Text Generation Models. *SIGIR 2018*.

---

## Appendix A — Full translationese-feature regex specifications

> [TBD] reproduce the regexes from `05_sft/translationese_scorer.py`.

## Appendix B — Full register suffix tables and priority order

> [TBD] reproduce the suffix tables from `05_sft/register_consistency.py`.

## Appendix C — Hand-labeled set descriptions

> [TBD] dataset card for Sets A, B, C, D, E. Released as a HuggingFace dataset.

## Appendix D — Reproducibility checklist

- Git commit: `8ff51d7` (frozen for this paper)
- Pinned requirements: `requirements.lock.txt`
- Three random seeds: 42, 1337, 2024
- All trained checkpoints released at [HF Hub URL TBD]
- All hand-labeled sets released at [URL TBD]
- All evaluation runs reproducible with `lm-evaluation-harness` commit `[TBD]`

---

## Placeholder summary (for the corresponding author)

This draft contains [XX.X] placeholders for numerical results from E1–E7. The
expected ranges are mid-points of the protocol's "Expected ranges" subsections.
To finalize:

1. Run E1–E7 per [EXPERIMENTAL_PROTOCOL.md](EXPERIMENTAL_PROTOCOL.md).
2. For each `[XX.X]` placeholder in this draft, replace with the measured
   value. Bold cells should be either bolded mean or bolded "best-in-column"
   per the table's convention.
3. Run `grep '\[' paper_draft.md` after substitution — only [TBD] author/affiliation
   placeholders should remain.
4. Decide final venue based on quality of measured results. If all KMMLU /
   KoBEST / HAERAE deltas are positive and ≥ 1.0 points absolute, target
   Knowledge-Based Systems or Information Processing & Management (SCI-Q1/Q2).
   If smaller, target IEEE Access or ETRI Journal (SCI-Q3/Q4).
