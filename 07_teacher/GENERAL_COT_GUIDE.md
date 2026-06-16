# Guide — General-Domain Chain-of-Thought Data Generation

A common confusion: "if I do RLVR on math and code, will my model naturally produce CoT
on any prompt?" The honest answer is **partially, but no — not enough alone**. This
guide walks through what the strong reasoning models actually do, what works for Korean
specifically, and how the four scripts in `07_teacher/` combine to fill the gap that
RLVR cannot.

## What R1 (and almost everyone since) actually did

The DeepSeek-R1 paper is the clearest published evidence on whether pure RLVR
generalizes to non-verifiable CoT:

| Pipeline | Did reasoning emerge? | Was the output shippable? |
|---|---|---|
| **R1-Zero** (base → pure RL on verifiable rewards) | yes | **no** — language mixing, poor readability, repetition |
| **R1** (base → cold-start SFT with curated CoT → RL → SFT → RL) | yes | yes |

The cold-start SFT data was the secret sauce. RL discovers *whether* reasoning is
correct; the cold-start SFT teaches *how* reasoning should be presented. Without the
cold-start phase, RL-only models develop reasoning that works but reads poorly —
which is exactly the failure mode you want to avoid in Korean where the base
distribution is weaker to begin with.

For your Korean case: pure RLVR would (1) reinforce English-CoT habits because the
base model's strongest reasoning is in English, (2) produce awkward Korean style
because the RL signal cares about correctness not naturalness, and (3) fail to
transfer to non-math prompts because algorithmic reasoning ≠ knowing why the
Mughal Empire declined.

## The four pillars of non-verifiable CoT data

For arbitrary prompts (advice, explanation, summary, creative, opinion, multi-step
analysis) there is no `verify()` function. The post-RLVR-paper consensus is that
**four techniques** between them cover the gap. Used together, they produce CoT
SFT and DPO data at the scale and quality the reasoning models ship with.

### 1. Distillation from a strong reasoner — `distill_from_teacher.py`

**The technique that actually scales.** A frontier reasoner (Claude with extended
thinking, GPT-4-class, DeepSeek-R1, Qwen3-235B-thinking) generates a CoT response
for any prompt you give it; you SFT your model on those (prompt, teacher-CoT) pairs.
Almost every open reasoning model released after R1 used this — many are *explicitly*
distilled from R1 (the "R1-Distill-*" family).

**Strengths:** quality bounded by the teacher (which is high), works for any prompt
domain, you control language by setting the system prompt, scales linearly with API
budget — 100k high-quality Korean CoT examples in a day is realistic.

**Costs and risks:** API spend is the dominant cost; teacher errors propagate
silently — always spot-check 200 random samples before scaling; some teachers
code-switch into English even when asked for Korean (the `acceptable_cot` filter
catches the most egregious cases via `korean_fraction` + structural CoT markers).

**Realistic teacher choices for Korean:**
- **Claude (Sonnet/Opus with extended thinking)** — excellent Korean, can be prompted into native CoT
- **DeepSeek-R1 / R1-Distill-32B** — open-weights, runs on a single 80GB H100, decent multilingual
- **GPT-4-class with reasoning** — strong but English-CoT bias even when asked otherwise
- **Qwen3-235B-Thinking** — open-weights, strongest Chinese/multilingual; Korean is decent

### 2. Self-consistency filtering — `self_consistency.py`

**Wang et al. 2022.** For any prompt, sample K responses from *your own model* at
high temperature. If multiple samples converge to the same answer, keep that group —
the consistency itself is evidence of correctness. If samples diverge, drop the prompt.

**Strengths:** no external teacher, no API cost, works on prompts where verifiable
rewards don't apply. The clustering uses lexical overlap of the conclusion section;
this catches near-paraphrases of the same answer while distinguishing genuinely
different answers.

**Limits:** the quality ceiling is your model's *own* ceiling. Self-consistency
doesn't make your model smarter — it makes its *confident* answers usable as SFT.
Use this after distillation has bootstrapped a baseline, not before.

**The right K:** 8 is the practical default. K=4 is the minimum that catches
hallucinations; K<3 is statistical noise; K>16 is rarely worth the compute.

### 3. AI-judge DPO — `ai_feedback_judge.py` (already in your package)

**For preference data on non-verifiable prompts.** Take two candidate responses
(from temperature sampling, or from before/after fine-tuning), ask the same model
or a stronger judge to compare them against a rubric, and emit `(prompt, chosen,
rejected)` as DPO training data.

The existing script already does the critical thing right: it judges in BOTH orders
and keeps only pairs where the two verdicts AGREE, cancelling position bias. That
filter is what turns this from a noisy heuristic into a usable signal.

**Pair this with distillation.** A common pattern: distill from teacher (SFT data) +
AI-judge two model outputs (DPO data). The DPO sharpens the model in places where
SFT plateaus.

### 4. Critique-and-revise — `critique_revise.py`

**Constitutional AI pattern (Bai et al. 2022).** The model produces an initial
response → the same model critiques it against a rubric → the same model revises
based on the critique. The (initial, revised) pair becomes a DPO `(rejected, chosen)`.

**Why it works:** for most prompts the same model is *better at finding flaws in
existing text than at avoiding those flaws in fresh generation*. Asking "what's
wrong with this answer?" extracts judgment the model wouldn't have applied in
one-shot generation.

**Failure modes the script guards against:**
- Empty critique ("the answer is good, no improvements needed") → drop the pair
  via `is_useful_critique()` keyword check
- Sycophantic revision that's identical to the initial → drop via
  `is_meaningful_revision()` Jaccard-distance check
- Revision drifts into English → reject via `korean_fraction()` gate

**Iteration depth:** one round of critique-revise is best. Two rounds usually
yields diminishing returns and three rounds tends to drift (the model agrees with
itself and the critique loop loses bite).

## A practical pipeline for Korean general CoT

Assuming you have a Stage-5 SFT model:

```
        ┌─────────────────────────────────────────────────────┐
Stage 5 │ SFT base model (your translated EN SFT + grounded)  │
        └─────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼───────────────────────────────────────┐
        │ A. distill_from_teacher.py                           │  60-80% of CoT data
        │    Use Claude/R1 to produce ~50k high-quality        │  comes from here
        │    Korean CoT examples across diverse prompt types    │
        └─────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼───────────────────────────────────────┐
        │ B. SFT on the distilled + 05_sft grounded data       │  the "cold-start"
        │    blend → "Stage 5b" instruct model with CoT habit  │  equivalent
        └─────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼───────────────────────────────────────┐
        │ C. Sharpen via two parallel paths:                   │
        │   - rejection_sample.py on verifiable tasks (math,   │
        │     code, MCQ, logic, format) → SFT + DPO            │
        │   - self_consistency.py on non-verifiable prompts    │
        │     → additional SFT                                 │
        │   - critique_revise.py on non-verifiable prompts     │
        │     → DPO pairs                                      │
        │   - ai_feedback_judge.py on candidate pairs          │
        │     → DPO pairs                                      │
        └─────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────▼───────────────────────────────────────┐
        │ D. Stage 6 DPO on the combined preference pool       │
        │    + Stage 8 GRPO on verifiable rewards              │
        │    → final reasoning model                           │
        └─────────────────────────────────────────────────────┘
```

This is the structure R1 (and most open reasoning models since) follows, with the
naming adjusted to fit our stages.

## Data volume — realistic targets

For an 8B base model adapting to Korean reasoning, the rough quantity floor:

| Source | Target | Why |
|---|---|---|
| `distill_from_teacher.py` (general) | 30-80k | the cold-start CoT pool; quality matters more than volume |
| `rejection_sample.py --verifier math` | 50-200k | verifiable signal is cheap, scale up |
| `rejection_sample.py` (code/mcq/logic/format combined) | 50-100k | same |
| `self_consistency.py` (general) | 10-30k | iterative self-distillation after the cold-start works |
| `critique_revise.py` (DPO) | 20-50k | preference data for non-verifiable |
| `ai_feedback_judge.py` (DPO) | 20-50k | same |
| **Total SFT** | **~150-400k** | enough for stable reasoning training |
| **Total DPO** | **~50-150k pairs** | enough to sharpen on non-verifiable |

These numbers are roughly proportional to what R1 used (curated cold-start in the
thousands, then orders of magnitude more from the RL + distillation loop). Scale
down if budget-limited — 30k high-quality distilled examples can take a model
surprisingly far on their own.

## Diversity and the "domain mix" question

**Cold-start data must be diverse.** R1's cold-start SFT covered math, code,
science, history, advice, creative writing, etc. — explicitly diverse. If your
distillation is all math (because that's what the teacher is best at), your model
will produce math-style CoT for everything, including "write me a wedding speech."
That's the canonical failure mode.

A good target distribution for the distilled CoT pool:

| Category | % | Notes |
|---|---|---|
| Math word problems | 15% | overlap with verifiable RFT; here for *style* |
| Code explanation / generation | 10% | overlap with verifiable RFT; here for style |
| Factual Q&A (history, science, geography) | 20% | Korean-context heavy |
| Analytical explanation ("why does X happen?") | 15% | the prototypical "needs CoT" category |
| Advice / planning ("how should I X?") | 10% | step-by-step structure naturally fits |
| Comparison ("compare X and Y") | 10% | structured CoT helps |
| Summarization / paraphrase | 5% | shorter CoT; mostly for variety |
| Creative writing | 5% | CoT is light here; show "I'm thinking about plot..." |
| Multi-step instruction following | 10% | structured response important |

Sources for the prompt pool that feeds the distiller: your `seed_instructions.json`
from Self-Instruct, the Korean-translated user turns from any English chat corpus,
and `prompts.jsonl` from your DPO pipeline. These are all already pools you have.

## Honest caveats

**1. The teacher is the ceiling.** If you distill from a teacher that's mediocre
at Korean, your student will be too. The distillation pipeline does not magically
improve quality beyond the teacher's level. Pick the strongest Korean reasoner you
have access to.

**2. Pure RLVR transfer is real but limited.** R1-Zero proved that some general
reasoning behavior emerges from pure RL — self-correction, step-by-step structure,
"wait, let me reconsider" moments. So if you literally have no other option, RLVR
alone *will* produce a model that does *some* CoT on general prompts. The output
will be readable but stylistically off, will have language-mixing issues in Korean,
and will not match what a distillation-trained equivalent achieves. It's a
backup plan, not the path.

**3. Self-consistency has a ceiling at your model's own ceiling.** It cannot
discover correct answers your model doesn't already know with some probability.
Use it after distillation has raised the ceiling.

**4. Critique-revise can collapse over rounds.** The model starts agreeing with
itself, the critiques become empty, and the technique stops producing useful
gradient. One pass per prompt is best; if you want more data, sample more prompts
rather than iterating more rounds on the same prompt.

**5. There is no shortcut for diversity.** Whatever your prompt pool looks like,
the resulting CoT data inherits its distribution. Curate the prompts; the rest
follows.
