# Guide — Data Mix and Training Order: Direct-Answer vs CoT

The most foundational question in setting up the training pipeline: **should I
train direct-answer capability first and then add CoT, or mix them together?**
And practically: how do I produce CoT-style grounded data at Stage 5, when the
default grounded generator only makes short direct answers?

## The short answer

**Mix from the start. Sequential training is actively worse.** Every production
reasoning model — R1, Qwen3, Claude, GPT-4 — trains direct-answer and CoT data
in the same SFT phase, with proportions reflecting which style each task
naturally calls for. The "sequential" approach (pure direct-answer SFT → then
pure CoT SFT) creates concrete problems that mixed training avoids.

## Why sequential is worse, in three concrete failure modes

### (1) Catastrophic forgetting of the direct style

If you SFT a model purely on direct answers and then SFT _again_ purely on CoT,
the second phase shifts the model's distribution toward verbose reasoning. The
model loses its ability to give a one-sentence answer to "조선민주주의인민공화국의 수도는
어디입니까?" — it will produce 200 words of reasoning to arrive at "평양".

This is annoying for users and wasteful of inference compute. Mixed training
prevents it because at every step the model sees both styles and learns to
preserve both.

### (2) Style selection is itself a learned behavior

Different prompts demand different response styles:

| Prompt                                            | Right style                        |
| ------------------------------------------------- | ---------------------------------- |
| "2 + 2 는 무엇입니까?"                            | Direct: "4"                        |
| "상인이 반지 48개를 팔고 그 절반을 더 팔았다 ..." | Multi-step CoT                     |
| "할머니가 외로워하시는데 어떻게 할가요?"          | Structured advice, light reasoning |
| "짧은 시를 한편 쓰시오"                           | Creative, no CoT                   |
| "세계 7대기적은 무엇입니까?"                      | Direct list                        |
| "로마제국은 왜 멸망했습니까?"                     | Analytical CoT                     |

A model that has only seen one style cannot route appropriately. The choice
"this prompt needs CoT" or "this prompt does not" is a _behavior the model has
to learn_, and learning it requires exposure to both styles in the same SFT
distribution. Sequential training never gives the model both styles
_simultaneously_, so it can't learn the routing.

### (3) The explicit evidence from Qwen3

Qwen3 ships with `/think` and `/no_think` modes — explicit toggles in the chat
template that turn CoT on or off. Both modes are trained in the **same** model
from the **same** SFT pool, with the system prompt switching between them. The
fact that this works at all is direct evidence that mixed training works; the
fact that Alibaba's most recent model line uses this design rather than two
separate models is evidence that mixed is the _better_ approach.

## The right proportion at Stage 5

Roughly, aggregating across the SFT mix:

| Task category                                           | Direct vs CoT                                  | Why                                   |
| ------------------------------------------------------- | ---------------------------------------------- | ------------------------------------- |
| Factual Q&A, definitions, titles, simple translation    | ~95% direct                                    | reasoning would be annoying overhead  |
| Summarization, fact extraction, outline                 | ~80% direct, 20% light CoT                     | structure helps; full CoT is overkill |
| Comparison, analytical explanation, "why" questions     | ~30% direct, 70% CoT                           | reasoning is the value                |
| Math word problems, multi-step reasoning, logic puzzles | ~5% direct, 95% CoT                            | reasoning is essentially required     |
| Code generation, debugging                              | ~40% direct (snippets), 60% CoT (explanations) | depends on complexity                 |
| Creative writing (story, poem)                          | ~95% direct                                    | CoT would damage the output           |
| Advice / planning                                       | ~50/50                                         | depends on prompt complexity          |

Aggregated for a general-purpose chat model, this comes out to roughly
**70-75% direct, 25-30% CoT**. R1's published mix is in this range; Qwen3's is
similar. If your distribution falls outside this band (e.g. 100% direct, or
100% CoT) you have a problem.

## How to produce grounded CoT data at Stage 5

Your default grounded generator (`05_sft/generate_grounded_ko.py`) has 9 task
types. **Eight of them produce direct-style data:**

`qa`, `multi_qa`, `summary`, `title`, `fact_extraction`, `explain_simply`,
`outline`, `definition`

**One task type, `reasoned_qa`, produces grounded CoT data.** It's the addition
that closes the gap your question identified. The difference from `qa`:

|                  | `qa`                     | `reasoned_qa`                       |
| ---------------- | ------------------------ | ----------------------------------- |
| Question type    | extractive: "What is X?" | inferential: "Why might X cause Y?" |
| Answer style     | 1-2 sentences, direct    | multi-step CoT grounded in passage  |
| Exemplar file    | `exemplars_qa.json`      | `exemplars_reasoned_qa.json`        |
| Trained behavior | short direct response    | step-by-step analytical response    |

A `reasoned_qa` exemplar looks like:

```json
{
  "단락": "대기오염은 호흡기질환을 일으킨다 ... 인디아와 중국의 상황이 가장 심각하다 ...",
  "질문": "발전도상나라의 대기오염사망률이 선진국보다 높을수 있는 리유는 무엇입니까?",
  "풀이와 답": "단계 1: 단락에 따르면 인디아와 중국의 대기질이 가장 나쁘고 이들은 발전도상나라이다.\n단계 2: 주요오염원은 차량과 산업이다.\n따라서 의료접근성이 낮은 발전도상나라에서 사망률이 더 높을수 있다."
}
```

The shipped `exemplars_reasoned_qa.json` has four such exemplars covering
cause-and-effect (environment), archaeological inference (Indus Valley), myth-
debunking (trees at night), and applied math (population growth).

Run it the same way as the other tasks:

```bash
python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf \
    --passages ko_passages.jsonl \
    --exemplars 05_sft/exemplars_reasoned_qa.json \
    --task reasoned_qa --out ko_grounded_cot.jsonl
```

## Where all your Stage-5 data comes from, in totality

The complete SFT data pool at Stage 5 typically draws from **five sources** —
some yield direct-style data, some CoT-style. The mix across all five is what
gives you the right 70/30 (ish) distribution:

| #   | Source                                                       | Style                    | Typical fraction |
| --- | ------------------------------------------------------------ | ------------------------ | ---------------- |
| 1   | Translated EN SFT (Alpaca, OpenOrca, etc.)                   | mixed, depends on source | 30-50%           |
| 2   | Grounded direct (`qa`, `summary`, `title`, `multi_qa`, etc.) | direct                   | 20-30%           |
| 3   | Grounded reasoning (`reasoned_qa`)                           | CoT                      | 10-15%           |
| 4   | Math/code CoT from Stage 7 bootstrap (fed BACK into Stage 5) | CoT                      | 10-20%           |
| 5   | Distilled CoT from a strong teacher (if available; Stage 7)  | CoT                      | 10-30%           |

Sources 4 and 5 close the loop you noticed: **Stage 7 is not a strictly later
stage; its outputs feed back into Stage 5 SFT.** This is why the script flow
in the main README shows iteration: produce Stage 5 SFT data, train, use that
model to drive Stage 7 generation, fold Stage 7's CoT data back into the SFT
pool, retrain. R1 ran 3-4 iterations of this loop; Qwen3 reportedly ran more.

## Where Stage 7 and Stage 8 fit in

Now the confusion clears up: **Stages 7 and 8 don't introduce CoT for the first
time — they enhance the CoT side of an already-mixed SFT model.**

- **Stage 5 (initial SFT)**: a balanced mix of direct + CoT, both styles
  represented. After this, your model can do both styles imperfectly. Solve
  rate on hard math problems might be 15-25%; analytical Q&A is shallow but
  present.

- **Stage 6 (DPO)**: preference data — chosen vs. rejected. Doesn't change
  _what_ the model can do, but sharpens _which_ of two candidates it prefers.
  Use AI-judge for non-verifiable, build_preference_data for the rest.

- **Stage 7 (teacher loop)**: enhances data volume across all categories. The
  bootstrap script (`few_shot_cot_bootstrap.py`) drives up CoT capability on
  verifiable tasks; `distill_from_teacher.py` adds general-domain CoT;
  `self_consistency.py` and `critique_revise.py` add medium-quality data
  without external teachers. All of these _fold back into Stage 5_ — they're
  data generators, not separate training stages.

- **Stage 8 (RLVR)**: pure RL on verifiable rewards. This sharpens reasoning
  on math/code/MCQ/logic/format where verifiers fire. It's the only stage that
  doesn't simply produce data — it does gradient updates with no SFT loss term.

## The honest order of operations

**Round 1** (everything from scratch):

1. Generate Stage 5 direct-style grounded data (8 task types except `reasoned_qa`).
2. Generate Stage 5 CoT-style grounded data (`reasoned_qa`).
3. Translate the English SFT corpus to Korean (Stage 5 source 1).
4. SFT on the combined pool → "Round-1 SFT model".

**Round 2** (using your Round-1 model to generate more data):

1. Run `few_shot_cot_bootstrap.py` against the BASE model with math problems →
   collect verified CoT chains for math.
2. Run `distill_from_teacher.py` (if you have API access) for diverse general
   CoT data.
3. Run `rejection_sample.py` against your Round-1 SFT model with verifiable
   problems → more verified CoT.
4. Combine: Round-1 SFT pool + Round-2 generated data → "Round-2 SFT pool".
5. Re-SFT from base on the Round-2 pool → "Round-2 SFT model".

**Round 3+**: same loop. Each round the model gets stronger, so the generation
step produces more correct/diverse data, so the next round's SFT pool is
larger and higher-quality.

**Stage 6 DPO and Stage 8 GRPO** run on top of any version of the SFT model.
They don't restart the loop; they sharpen whatever SFT model you give them.

## What you don't need to do

A few things people sometimes assume but shouldn't:

- **You don't need separate models for direct vs CoT.** Same model, mixed data.
- **You don't need to flag every example as direct-vs-CoT in the training data.**
  The model figures out the style from the (prompt, response) pairing.
- **You don't need to train direct first to "establish a baseline".** That's the
  sequential trap; just mix from the start.
- **You don't need `reasoned_qa` to dominate the grounded mix.** It's the
  smaller share — your passage pool should produce 4-5 direct examples per
  passage (qa/summary/title/etc.) and 1 reasoned_qa example per passage. The
  ratio reflects how often each style is naturally called for.

## Pitfalls if you get the proportions wrong

- **Too much CoT (>50% of SFT pool)**: model becomes verbose on simple
  questions, slow at inference, annoying for everyday use.
- **Too little CoT (<10% of SFT pool)**: model never thinks step-by-step even
  when prompted explicitly with "let's think step by step" — the behavior is
  too rare in training to surface reliably.
- **CoT only in math, no CoT in general prompts**: model can reason about math
  but produces shallow answers for analytical questions. This is exactly the
  failure mode `reasoned_qa` exists to prevent — broaden the CoT distribution
  beyond verifiable domains.
- **CoT data with English-CoT-Korean-answer mixing**: the model imitates this
  and code-switches. Filter aggressively with `cjk_fraction` / `korean_fraction`
  on the CoT portion (not just the final answer).

## Summary in one sentence

Mix direct-answer and CoT data in your Stage 5 SFT pool from the very first
round, use `reasoned_qa` to close the grounded-CoT gap that the other task
types don't cover, and treat Stages 7 and 8 as _enhancement_ loops that fold
their outputs back into Stage 5 rather than as separate "add CoT later" steps.
