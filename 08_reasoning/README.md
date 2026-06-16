# Stage 8 — RLVR and Korean Chain-of-Thought

## The core insight

For reasoning tasks (math, code, logic), **correctness is language-independent**.
A correct numerical answer is correct whether the reasoning was in English, Korean,
or Klingon. This is the entire reason RLVR works for low-resource languages:
you don't need a strong native judge, you don't need a reward model, you don't
need annotators — you have **arithmetic** (or unit tests, or a checker), and that
signal can't be gamed by the model's own judgment.

This stage exploits that fact to produce Korean chain-of-thought data and to train
the model with RL even when its Korean is still developing.

## The pipeline

```
English math dataset            verify_math.py        ←─────────── language-independent
  (GSM8K, MATH, …)                  ↑                              correctness signal
       │                            │
       │  translate_problems.py     │  reject + emit
       ▼                            │
  Korean problems    ─►  rejection_sample.py  ─►  ko_math_cot.jsonl  (Korean CoT SFT)
  {problem, answer}      (sample K, verify)        + ko_math_dpo.jsonl  (optional)
                                    │
                                    ▼
                              grpo_train.py  ─►  RLVR-trained model
                              (uses the same verifier as REWARD)
```

Each script and what it does:

| Script                  | Role                                                                     | Pure logic?                 |
| ----------------------- | ------------------------------------------------------------------------ | --------------------------- |
| `verify_math.py`        | language-independent verifier (answer extraction + numeric compare)      | yes — selftest              |
| `translate_problems.py` | translate English problems → Korean, gold answer passes through          | reuses `md_translate.py`    |
| `rejection_sample.py`   | sample K, verify, emit verified Korean CoT as SFT (+ optional DPO pairs) | drives `chat()` hook        |
| `grpo_train.py`         | template for GRPO with the verifier as the reward function               | TRL + Megatron-Bridge paths |

## Two ways to use RLVR

### (1) Rejection-sampling fine-tuning (RFT) — the cheap path

Run `rejection_sample.py`: for each problem, sample K solutions, keep the verified
ones, train via plain SFT on (problem, verified-CoT). No RL machinery needed; just
SFT on a filtered subset. This is what "STaR" / "ReSTEM" do. It almost always
beats no RFT and is usually 80% of what RL gets you, at 10% of the engineering.

### (2) GRPO (the "R1-style" recipe) — the expensive but better path

Run `grpo_train.py`: for each prompt, sample G completions, score each with the
verifier, use the within-group relative advantage as the policy gradient signal.
Group Relative Policy Optimization (Shao et al.) is what DeepSeek-R1 used; no
reward model needed because the verifier IS the reward.

Start with (1); add (2) once (1) plateaus.

## Where the Korean CoT data comes from

This is the subtle and important part. The model's chain-of-thought is in
**whatever language it samples in**. Two failure modes worth knowing:

**Code-switching CoT.** A freshly-adapted Korean model often reasons partly in
English, especially for math. Force a Korean system prompt ("당신은 수학교원
입니다 ... 단계별로 조선어로 푸시오") and add a soft filter post-hoc that
rejects samples where the CoT Hangul fraction is below threshold, even if the
final answer is correct. Otherwise you SFT on English-CoT-with-Korean-veneer.

**Reasoning quality vs. language.** Sometimes letting the model reason in English
internally gives much better correctness, even if the final answer is in Korean.
The pragmatic move: in your first RFT pass, accept English CoT (correctness
matters most when bootstrapping). Once the verified pool is large, do a second
pass restricted to Korean-CoT samples to lift the language quality of reasoning.
Train on the union.

## Where to get the source problems

You don't have a Korean math dataset, so translate. Recommended sources, all
English, all freely available:

| Dataset    |  Size | Difficulty   | Notes                                           |
| ---------- | ----: | ------------ | ----------------------------------------------- |
| GSM8K      |  8.5k | grade-school | The standard, easiest to verify                 |
| MATH       | 12.5k | competition  | `\boxed{}` answer convention — verifier handles |
| MathQA     |   37k | varied       | multiple-choice; extract the chosen letter      |
| MetaMathQA | ~400k | grade-school | GSM8K augmented; great for volume               |
| Big-Math   | ~250k | mixed        | filtered, deduplicated; good general pool       |

Translate via `translate_problems.py` (which reuses our markdown/LaTeX-preserving
translator so numbers and equations survive intact). Hold out a few thousand
problems for evaluation; never put eval problems into training.

## Beyond math: code and other verifiable domains

The same RLVR machinery extends to anything with an automatic checker, and the
package now ships **five** verifiers, all conforming to the same one-line interface:

```python
verify(model_output: str, gold) -> bool
```

| Verifier           | What it checks                                              | Gold format                                                                                                                                          |
| ------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `verify_math.py`   | numeric answer extraction + tolerant equality               | a number / numeric string                                                                                                                            |
| `verify_code.py`   | extracts code, runs unit tests in a subprocess with timeout | a string of test code OR `{"tests":..,"timeout":..}`                                                                                                 |
| `verify_mcq.py`    | multiple-choice label extraction (A-E / 가-마 / ①-⑤ / 1-5)  | a label "A".."E" / "가".."마" / "①".."⑤" OR `{"answer":..,"n_options":..}`                                                                           |
| `verify_logic.py`  | Sudoku + a generic constraint-satisfaction framework        | `{"type":"sudoku","puzzle":[[...]]}` or `{"type":"assignment", "keys":[..], "constraints":[..]}`                                                     |
| `verify_format.py` | JSON / list / regex / markdown-table structure following    | `{"type":"json","schema":{..}}`, `{"type":"list","min_items":3}`, `{"type":"regex","pattern":..}`, `{"type":"md_table","columns":[..],"min_rows":N}` |

**Plug them into the pipeline via the `--verifier` flag:**

```bash
# math (default)
python rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_math.jsonl \
    --out-sft ko_math_cot.jsonl --verifier math

# code: input {problem, gold} where gold = unit-test code string
python rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_code.jsonl \
    --out-sft ko_code_cot.jsonl --verifier code

# mcq: input {problem, gold} where gold = "A" / "B" / ...
python rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_mcq.jsonl \
    --out-sft ko_mcq_cot.jsonl --verifier mcq

# logic (sudoku): input {problem, gold:{"type":"sudoku","puzzle":[[...]]}}
python rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_sudoku.jsonl \
    --out-sft ko_sudoku_cot.jsonl --verifier logic

# format following: input {problem, gold:{"type":"json","schema":{..}}}
python rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_struct.jsonl \
    --out-sft ko_struct_cot.jsonl --verifier format
```

Each verifier auto-picks a task-appropriate Korean system prompt; override with
`--system` if needed. The rest of the pipeline (`grpo_train.py`) plugs in the
same way — swap the `verify_math` import for whichever verifier you need; the
reward function shape is identical.

### Adding a new verifier

Drop a `verify_<name>.py` next to the others, expose a `verify(output, gold) -> bool`
function (plus your own `extract_*` helpers), include a `--selftest` block, and
add `"<name>"` to `--verifier`'s choices in `rejection_sample.py`. That's all the
contract.

### Per-verifier notes worth knowing

**`verify_code`**: the subprocess runs untrusted model code with a timeout but
_no resource isolation_. For production scale, wrap the subprocess in firejail /
gVisor / nsjail / docker, block network, and limit memory. The timeout protects
against infinite loops but not against malicious imports or file writes.

**`verify_mcq`**: handles Korean labels (가/나/다/라/마 ↔ A/B/C/D/E, circled ①-⑤, digits) and picks the
LAST marker in the response — models often say "I considered A but… the answer
is C", and you want C.

**`verify_logic`**: Sudoku is shipped concretely; for other puzzles, write an
`extract_<puzzle>` function and a list of constraints, then call
`verify_constraints(extracted, constraints)`. Constraint strings (used in the
generic `assignment` type) `eval` against a safe-builtins namespace.

**`verify_format`**: JSON extraction is forgiving (tries fenced blocks first, then
longest balanced `{...}` / `[...]`, then the whole response). Schema check is a
tiny built-in (required keys + value types) to avoid a `jsonschema` dependency;
swap to the real library if you need full Draft-7+ support.

## Practical hyperparameters

For `rejection_sample.py`:

- **K (samples per problem):** 8 is a good default. Higher K helps on harder
  problems but costs linearly more compute. Drop to 4 once your model's solve
  rate is high.
- **Temperature:** 0.7-0.9. Too low → all samples identical, no benefit from K.
- **`--keep-per-problem`:** 1-2. Keeping ALL correct samples over-weights
  easy problems where K=8 might all be correct.

For `grpo_train.py`:

- **Group size G:** 4-8. R1 used 16; smaller is fine for budget.
- **Learning rate:** 5e-6 (RL is sensitive to LR; lower than SFT).
- **Reward shaping:** the example adds a tiny format bonus (0.1) for using the
  Korean answer marker. Keep shaping weights << correctness weight (1.0) so they
  don't dominate.

## What this stage produces

After Stage 8 you have:

- Verified Korean CoT SFT data (`ko_math_cot.jsonl`) — fold back into Stage 5.
- Optional DPO pairs from same-problem (verified, unverified) → Stage 6 input.
- An RLVR-trained model that reasons better in Korean.

And, because the verifier is reusable, you can run more iterations as the model
improves: new model solves more problems → larger verified pool → better next-iter
training. This is the cleanest, lowest-risk loop in the whole pipeline.
