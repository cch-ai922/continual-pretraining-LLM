# Guide — Bootstrapping CoT From the BASE Model (No External Teacher)

The honest historical answer to "what did the original LLM developers do when no
strong reasoner existed to distill from?" is **STaR** — Self-Taught Reasoner,
Zelikman et al. 2022. This guide explains the technique, why it works, when to
use it, and how the scripts in this directory implement it for Korean.

## The chicken-and-egg problem isn't real

A natural worry: "if every modern reasoning model was distilled from a stronger
one, where did the _first_ one come from?" The answer is that **pretraining
itself contains reasoning data**. Every base model has seen:

- Math textbooks with worked examples
- math.stackexchange and other Q&A sites with step-by-step explanations
- GitHub READMEs with reasoning walkthroughs
- Scientific papers with proofs
- Khan Academy / educational sites
- Code with comments explaining algorithms
- Wikipedia articles with derivations
- Tutoring transcripts on the open web

The base model **knows what reasoning looks like**. It has internalized the
_forms_ of step-by-step argument. It just doesn't reliably _produce_ them on
demand — its default behavior is to mimic whatever distribution dominates its
recent context, which for most prompts is short answers.

The original bootstrap unlocks the latent capability. There's no chicken-and-egg
— the egg was laid during pretraining.

## STaR in five steps

| Step | What happens                                                   | Required tooling                        |
| ---- | -------------------------------------------------------------- | --------------------------------------- |
| 1    | Few-shot CoT prompt the base model with hand-written exemplars | `exemplars_math_cot.json`               |
| 2    | Sample K continuations at temperature ≥ 0.7                    | base model with completion API          |
| 3    | Verify each continuation against the gold answer               | `08_reasoning/verify_*.py`              |
| 4    | KEEP the correct continuations as SFT data                     | `few_shot_cot_bootstrap.py`             |
| 5    | (Optional) Rationalize the failures                            | `--rationalize` flag in the same script |

Then: SFT the base model on the kept data, and **run the whole loop again**.
After 1-2 iterations the model's solve rate goes up substantially because:

- It now produces CoT structure reliably (the SFT taught the form)
- The temperature sampling discovers correct paths the SFT model wouldn't reach greedily
- More problems become solvable; the kept pool grows; the next SFT round is better

## Rationalization — the under-appreciated trick

This is the part most people miss. STaR's contribution wasn't sampling-and-filtering;
that was already known. It was _what to do with the failures_.

**The asymmetry**: forward reasoning ("find the answer") is hard. Backward
rationalization ("given the answer, explain how to get there") is _much easier_
— for the same reason it's easier to verify a proof than to discover one.

So when 0 of K samples solve a problem forward, the script instead asks:

> "We know the answer is X. Walk through the steps that lead to X."

The model is much more likely to produce a valid chain in this mode because the
target is given. We still verify the chain — sometimes the model produces
plausible-looking reasoning that doesn't actually reach the stated answer — but
the success rate on rationalized chains is typically 40-70%, compared to maybe
10-30% on forward solving for the same hard problems.

**Why this matters**: without rationalization, you only collect training data on
problems the model can already solve once-in-K. That biases the curriculum
toward the easy tail — you don't learn anything from problems just outside the
model's current reach. Rationalization captures the "almost solved" problems and
turns them into training data. The next iteration's model can then solve some
of those forward.

## Combining bootstrap with the rest of the pipeline

The bootstrap script and the distillation script (`distill_from_teacher.py`) are
**not exclusive** — they're alternatives based on what resources you have:

| Have a strong teacher API? | Use                                                                |
| -------------------------- | ------------------------------------------------------------------ |
| Yes (Claude, R1, GPT-4)    | `distill_from_teacher.py` — faster, higher quality                 |
| No, only your base model   | `few_shot_cot_bootstrap.py` — slower, quality bounded by base      |
| Yes but limited budget     | Use distillation for 10-30k seed examples, then bootstrap to scale |

The bootstrap-only path takes longer to reach the same quality, but it works and
costs only your own compute. For verifiable domains it works _well_ — the
verifier provides clean signal. For non-verifiable domains it's harder; see the
section below.

## A complete bootstrap-only pipeline for Korean

Starting from your Stage-4 base model (no SFT yet, just continual-pretrained on Korean):

```bash
# Round 1: extract latent reasoning via STaR
python 07_teacher/few_shot_cot_bootstrap.py \
    --model ./qwen3-ko-base-hf \
    --exemplars 07_teacher/exemplars_math_cot.json \
    --problems ko_gsm8k_train.jsonl \
    --out-sft ko_cot_r1.jsonl \
    --verifier math --k 16 --rationalize

# Round 1 SFT on the verified chains
python 05_sft/sft_train.py --data ko_cot_r1.jsonl --out ./qwen3-ko-bootstrap-r1

# Round 2: rerun bootstrap on the same problems with the improved model
python 07_teacher/few_shot_cot_bootstrap.py \
    --model ./qwen3-ko-bootstrap-r1 \
    --exemplars 07_teacher/exemplars_math_cot.json \
    --problems ko_gsm8k_train.jsonl \
    --out-sft ko_cot_r2.jsonl \
    --verifier math --k 8 --rationalize    # smaller K because solve rate is up

# Round 2 SFT on rounds 1+2 combined
cat ko_cot_r1.jsonl ko_cot_r2.jsonl > ko_cot_combined.jsonl
python 05_sft/sft_train.py --data ko_cot_combined.jsonl --out ./qwen3-ko-bootstrap-r2

# Round 3+: add the OTHER verifiers (code, mcq, format) — same script, different --verifier
python 07_teacher/few_shot_cot_bootstrap.py --verifier code \
    --problems ko_humaneval.jsonl --out-sft ko_code_cot.jsonl \
    --exemplars 07_teacher/exemplars_code_cot.json --model ./qwen3-ko-bootstrap-r2

# Then GRPO on top (Stage 8) for further sharpening — same model, same verifiers.
```

After this pipeline, you have a reasoning model with no external dependency on
any other LLM. This is what was done before strong teachers existed; it's still
the right path when API access is constrained.

## What about non-verifiable CoT in the no-teacher case?

This is the hard part. Without a verifier and without a teacher, the only signal
you have is the model's own consistency and self-judgment. The realistic options:

**1. Self-consistency (`self_consistency.py`).** Sample K, cluster by conclusion
similarity, keep the majority cluster. This works but quality is bounded by what
the model gets right by chance more than once-in-K. Use after the verifiable
bootstrap has raised the base capability.

**2. Critique-and-revise (`critique_revise.py`).** Same model produces, critiques,
revises. The asymmetry (easier to spot flaws than to avoid them) gives you DPO
pairs. Empirically the gains are real but modest — maybe 30-40% as effective as
DPO from a strong external judge.

**3. Cross-domain transfer.** This is the deepest answer to your question. The
_form_ of reasoning learned on verifiable tasks transfers to non-verifiable
tasks. A model that's been bootstrapped to produce step-by-step math CoT will,
when given a non-math prompt, often produce step-by-step structure for the new
prompt too. The transfer isn't perfect — math-style reasoning forced onto a
recipe prompt is awkward — but it's real. R1-Zero demonstrated this most
publicly: pure RL on verifiable tasks did produce some general CoT, the
reasoning was just stylistically off.

**The practical workflow for the no-teacher case** is therefore:

- Bootstrap heavily on verifiable tasks (math + code + MCQ + format + logic)
- The model develops a _habit_ of producing structured reasoning
- Add self-consistency + critique-revise rounds on diverse non-verifiable prompts
- Accept that the final quality on creative/advice/opinion will be lower than
  what distillation would produce, but it's still much better than no bootstrap

## What you give up vs. distillation

| Aspect                     | Bootstrap-only                          | Distillation              |
| -------------------------- | --------------------------------------- | ------------------------- |
| External cost              | $0 (your compute)                       | API fees                  |
| Quality ceiling            | Base model's latent ability             | Teacher's actual ability  |
| Time to first usable model | 3-5 bootstrap rounds                    | 1 distillation pass + SFT |
| Style quality              | Sometimes awkward; depends on exemplars | Inherits teacher's style  |
| Verifiable domain quality  | Excellent (verifier is the signal)      | Excellent                 |
| Non-verifiable quality     | Limited                                 | Excellent                 |
| Independence               | Total                                   | Tied to teacher           |

For a serious production model, the answer is usually "distillation if you can
afford it, bootstrap otherwise, and ideally both in combination." The history
question — what did the first reasoning models do? — is unambiguously the
bootstrap path. STaR is _how_ the eggs were originally laid.

## Tuning knobs that matter

- **K (samples per problem).** Higher K = more chances to find a correct chain =
  more training data, but linear cost. K=16 in round 1 is reasonable for hard
  problems; K=8 in later rounds is enough since the base model improves.
- **Temperature.** Too low (≤ 0.5) and all K samples are the same; too high (≥ 1.2)
  and they're incoherent. 0.7–0.9 is the sweet spot.
- **Number of exemplars.** 4-8 is the empirical sweet spot. More just bloats the
  context; fewer doesn't establish the pattern reliably.
- **`--keep-per-problem`.** Capping correct samples kept per problem (e.g. to 2)
  prevents easy problems from dominating the training set. Important.
- **`--rationalize` on or off.** Always on for the first 1-2 rounds — that's
  where the curriculum-extension matters most. Optional in later rounds when
  forward solve rate is already high.

## A note on exemplar quality

The `exemplars_math_cot.json` ships with 8 hand-written native-Korean CoT
exemplars covering: arithmetic, percentages, algebra, geometry, distance-speed-
time, multi-step, work problems, and consecutive integers. They follow a
consistent format:

```
단계 1: <first step>
단계 2: <second step>
...
최종답: <answer>
```

**This consistency is critical.** The model copies the _shape_ of the exemplars
verbatim, including the marker words ("단계", "최종 답"). If you change the
exemplars to use different markers, the generated outputs will use those markers
and your downstream `verify_math` will need to match.

For other verifiable domains, write parallel exemplar files:

- `exemplars_code_cot.json` — (problem, full solution with step-by-step comments, expected output)
- `exemplars_mcq_cot.json` — (question + choices, reasoning, "정답: X")
- `exemplars_logic_cot.json` — (puzzle, step-by-step deduction, final solution)

Then point the bootstrap script at the right exemplars + the right `--verifier`.

## The closing honest framing

If someone asks "did the original LLM developers really do this?" — yes,
demonstrably, this is the published technique. STaR was Stanford 2022. The same
ideas (sample-filter-train) drove Codex training, AlphaCode, and the early GPT
reasoning work. R1-Zero is essentially "STaR with RL instead of SFT for the
training step." The lineage is unbroken back to base models with no external
teacher whatsoever.

The reason distillation feels like the dominant story now is that as soon as
strong reasoners _did_ exist, they became the cheapest way for the next
generation to acquire reasoning data. That doesn't mean the original technique
stopped working — it means most labs took the cheaper path for newer models.
For your low-resource Korean case, the original technique is genuinely your best
option in the budget-constrained regime, and the script in this directory is
the same recipe Zelikman et al. published in 2022, adapted for native-Korean
exemplars and your 5-verifier framework.
