# Adapting Qwen3-8B to Korean — Full Pipeline

End-to-end recipe to take **Qwen3-8B** (strong multilingual base, weak Korean) and turn it into a
Korean-competent model that **retains English/knowledge**, then use the result as a **lever to
generate Korean SFT and DPO data**.

> ✅ **Architecture note — this is the clean path.**
> Qwen3-8B is a **standard dense decoder-only transformer** (no GDN, no MoE, no MTP, no vision):
>
> | field | value |
> |---|---|
> | layers | 36 |
> | hidden size | 4096 |
> | FFN (intermediate) | 12288 |
> | attention heads (Q) | 32 |
> | KV heads (GQA) | 8 |
> | head_dim | 128 |
> | norm | RMSNorm + **QK-Norm** |
> | activation | SwiGLU (silu) |
> | position | RoPE (θ≈1e6) |
> | vocab | 151,936 (BBPE, 119 langs) |
> | tied embeddings | **No** (init input AND lm_head) |
> | context | 32,768 native |
>
> Implications baked into this pipeline:
> 1. **Megatron-Bridge (Qwen3 provider) is recommended** for clean HF↔Megatron conversion; because
>    the architecture is standard, **vanilla Megatron-LM with a Qwen converter also works**. Either is fine.
>    Needs `transformers >= 4.51`.
> 2. **No vision / no MTP** → embedding surgery is just input embeddings + lm_head. Plain next-token CE loss.
> 3. **Tokenizer extension is more likely worthwhile than on Qwen3.5** (151k/119-lang vocab vs 248k/201).
>    Still gate the decision on measurement (`00_data/measure_fertility.py`), but expect a "yes" for Korean.
> 4. **Extend BPE correctly:** Qwen's tokenizer is byte-level BPE. Proper extension splices the new
>    pieces into the BPE **model** — union the vocab AND append the **merge rules** — so new tokens are
>    produced *compositionally* by BPE. Do **not** use `tokenizer.add_tokens()` (a literal added-token
>    side channel that bypasses the merge table) and do **not** train SentencePiece (wrong alphabet).
>    `01_tokenizer/extend_tokenizer.py` does the merge; `test_tokenizer_merge.py` proves it works.

### Alternative base: Qwen3.5-9B (hybrid GDN + early-fusion VL)

If you want a stronger base, **Qwen3.5-9B** is a defensible swap. It has more
latent capability (MMLU-Pro 82.5 vs Qwen3-8B ~71), 201-language coverage (vs
119), and 262K native context, but the architecture is fundamentally different:
hybrid 3:1 Gated DeltaNet / Gated Attention, 32 layers, 248K vocab, MTP head,
early-fusion vision-language (vision parameters baked in but ignorable at
text-only inference). The engineering cost is real — different Megatron-Bridge
provider, lower initial LR for GDN stability, `transformers >= 5.2.0` — but
Stages 5-8 of the pipeline are model-agnostic and work unchanged.

See **`QWEN35_MIGRATION_GUIDE.md`** for the full comparison and
**`04_pretrain/common_config_qwen35.py`** for the Stage-4 config dict.

---

## Stage overview

| Stage | Name | What it does | Scripts |
|------|------|--------------|---------|
| 0 | Data audit & blend | Measure Korean fertility; compute token budgets & sampling weights | `00_data/` |
| 1 | Tokenizer (likely yes) | Train byte-level Korean BPE; either **EXTEND** Qwen's BBPE (union vocab+merges, default) or **REPLACE** Korean tokens entirely (research path — see `TOKENIZER_REPLACEMENT_GUIDE.md`) | `01_tokenizer/` |
| 2 | Embedding init | **Averaging-init** for extension (subword-average new rows); **Random-init** for replacement (clean-slate Korean embeddings) — pair must match Stage 1 choice | `02_embeddings/` |
| 3 | Convert HF → Megatron | Bring (resized) model into Megatron-Core | `03_convert/` |
| 4a | Embedding-only warmup | Freeze body, train only embeddings so new tokens settle | `04_pretrain/stage1_embedding_only.py` |
| 4b | Continual pretrain | Full or LoRA+embeddings on Korean + English-replay + parallel | `04_pretrain/stage2_continual.py` |
| 4c | Long-context retain | Brief 32K-seq phase on genuinely long docs so native 32K context survives the 4096 continual run | `04_pretrain/stage2c_longctx.py` |
| 5 | SFT | Translate EN SFT → Korean + grounded native data (9 task types — 8 direct-style + `reasoned_qa` for grounded CoT); see `DATA_MIX_AND_ORDERING.md` for the direct-vs-CoT mix decision | `05_sft/` |
| 6 | DPO | Build preference pairs (verifiable signals) + train | `06_dpo/` |
| 7 | Teacher loop | Self-Instruct + AI-judge; **STaR-style bootstrap from base** (no external teacher) — see `07_teacher/SELF_BOOTSTRAP_GUIDE.md`; **distillation from a strong reasoner**, **self-consistency**, **critique-revise** for non-verifiable CoT — see `07_teacher/GENERAL_COT_GUIDE.md` | `07_teacher/` |
| 8 | RLVR / Korean CoT | Translate problems → reject-sample verified solutions → Korean CoT SFT + GRPO | `08_reasoning/` |
| 9 | Eval | Score Korean fluency, code-switching, verifiable accuracy, **and English regression** every checkpoint | `09_eval/` |

If fertility comes back acceptable, skip Stages 1, 2, and 4a and go straight to continual pretrain.

---

## Your data inventory & the math

| Source | Size | Notes |
|--------|------|-------|
| Korean monolingual | ~20B tokens | the binding constraint — small, so we upsample |
| English monolingual | ~400B tokens | huge — we only *replay* a thin, fresh slice |
| Parallel EN–KO | 25M pairs (~1.0–1.3B tokens) | cross-lingual alignment + knowledge transfer |
| English SFT | 2M examples | translated to Korean in Stage 5 |

**Where to get Korean data (suggested, license-permitting):**
- *Monolingual / corpora:* 모두의 코퍼스 (National Institute of Korean Language — Modu Corpus), AI Hub datasets, KLUE corpora, Korean Wikipedia, NamuWiki dumps, news/web crawls.
- *Parallel EN–KO:* AI Hub translation corpora, OPUS, Tatoeba.
- *Evaluation (Stage 9):* **KMMLU** (knowledge MCQ), **KoBEST**, **HAERAE-Bench**, **CLIcK** (Korean cultural/linguistic), plus your own frozen task set and an English-regression slice.
- Korean is comparatively well-covered by Qwen3's BBPE, so still gate tokenizer extension on `measure_fertility.py` rather than assuming it's needed.

### Continual-pretrain token budget & blend (Stage 4b default)

Target a **~60B-token** continual run (scale with compute):

| Source | Weight | Tokens seen | Epochs over source |
|--------|-------:|------------:|-------------------:|
| Korean mono | 45% | 27.0B | ~1.35× (good) |
| English replay | 45% | 27.0B | ~0.07× (fresh, diverse — ideal replay) |
| Parallel EN–KO | 10% | 6.0B | ~4.6× (watch for memorization) |

Korean gets the most exposure (we're *teaching* it); English is replayed only enough to prevent
forgetting (the base is already excellent at it); parallel data is the alignment bridge that carries
English knowledge into Korean.

**Tuning knobs:** English regresses → raise English weight (45→55%) or lower LR. Parallel memorizes
(rising train acc, flat eval) → drop parallel to 6–8% or dedup/augment; keep any single source under
~4 epochs. Korean stalls → raise Korean weight or extend the budget; don't starve English to do it.

### Stage 4a (embedding-only) blend — Korean-heavy, small budget (~2–4B tokens)
Korean 70% / Parallel 20% / English 10%.

---

## Hyperparameters (defaults — tune to your cluster)

| | Stage 4a (emb-only) | Stage 4b (continual) | Stage 5 (SFT) | Stage 6 (DPO) |
|--|--|--|--|--|
| Trainable | embeddings + lm_head only | full **or** LoRA + embeddings | full or LoRA | LoRA |
| Peak LR | 1e-4 | **2e-5** (low!) | 1e-5 | 5e-6 |
| Schedule | cosine, 2% warmup → 10% | cosine, **re-warmup** 1–2% → 10% | cosine, 3% warmup | cosine |
| Seq len | 4096 | 4096 | 4096 (packed) | 2048 |
| Precision | bf16 | bf16 | bf16 | bf16 |
| Global batch (tokens) | ~1M | ~4M | ~0.5M | — |

The single most important number is the **Stage 4b peak LR** (~2e-5 with a short re-warmup). Too high
here is the classic way to nuke the English ability you're trying to keep (cf. Ibrahim et al. 2024,
"Simple and Scalable Strategies to Continually Pre-train LLMs": re-warm, re-decay, + replay).

### Parallelism (per node of 8×H100, 8B dense)
`TP=2, PP=1, CP=1, DP=<rest>` is comfortable for 8B (a standard dense model fits easily). Bump TP to 4
only if activation memory at seq 4096 is tight. No recurrent-state caveats (unlike GDN models).

---

## Run order

```bash
# Stage 0 — audit (ALWAYS first)
python 00_data/measure_fertility.py --model Qwen/Qwen3-8B --korean-text data/ko_sample.txt \
    --compare-bpe ko_bpe.json

# Stage 1+2 — extend tokenizer (likely worthwhile for Qwen3 + Korean)
python 01_tokenizer/train_korean_bpe.py   --corpus data/ko_corpus.txt --base Qwen/Qwen3-8B --vocab-size 16000 --out ko_bpe.json
python 01_tokenizer/extend_tokenizer.py  --base Qwen/Qwen3-8B --korean-bpe ko_bpe.json --out ./qwen3-ko-tok
python 02_embeddings/init_new_embeddings.py --model Qwen/Qwen3-8B --new-tokenizer ./qwen3-ko-tok --out ./qwen3-ko-init

# Stage 0 (cont.) — build the Megatron data blend
python 00_data/build_data_mix.py --budget-b 60
bash   00_data/preprocess_pretrain.sh

# Stage 3 — HF -> Megatron
bash 03_convert/hf_to_megatron.sh ./qwen3-ko-init ./mcore_ckpt   # or Qwen/Qwen3-8B if not extended

# Stage 4 — continual pretrain
python 04_pretrain/stage1_embedding_only.py    # optional, only if extended
python 04_pretrain/stage2_continual.py
# Stage 4c — keep native 32K context alive (Qwen3-8B is native 32K; the 4096
# continual run above never trains positions 4096..32768). Prep a long-doc bucket
# and run a brief 32K retention phase from the Stage-4b checkpoint:
python 00_data/build_long_docs.py --in ko_articles.jsonl --out ko_long.jsonl \
    --group-key topic --target-tokens 28000 --tokenizer ./qwen3-ko-base-hf   # + same for English
# preprocess ko_long.jsonl/en_long.jsonl to *_long_text_document, then:
python 00_data/build_data_mix.py --budget-b 60 --long-budget-b 4   # also writes blend_long.yaml
python 04_pretrain/stage2c_longctx.py

# Stage 3 (back) — Megatron -> HF
bash 03_convert/megatron_to_hf.sh ./mcore_ckpt_final ./qwen3-ko-base-hf

# Stage 5 — SFT (reuses ../md_translate.py)
python 05_sft/translate_en_sft_to_ko.py --in en_sft.jsonl --out ko_sft.jsonl --lid-check
# passage-grounded NATIVE data from the BASE model via few-shot (see exemplars_qa.json):
python 05_sft/generate_grounded_ko.py --model ./qwen3-ko-base-hf --passages ko_passages.jsonl \
    --exemplars 05_sft/exemplars_qa.json --task qa --out native_raw.jsonl
# filter grounded data with the BASE model (PMI + few-shot NLI faithfulness):
python 05_sft/faithfulness_scorer.py --model ./qwen3-ko-base-hf --in native_raw.jsonl --out native.jsonl
python 05_sft/build_sft_blend.py --translated-hi ko_sft.jsonl --native-hi native.jsonl --english en_sft.jsonl --out ko_sft_blend.jsonl
python 05_sft/sft_train.py

# Stage 6 — DPO
python 06_dpo/build_preference_data.py --prompts prompts.jsonl --out ko_dpo_pairs.jsonl
python 06_dpo/dpo_train.py

# Stage 7 — teacher loop (after you have the Stage-5 instruct model): make MORE data
python 07_teacher/self_instruct.py --model ./qwen3-ko-sft-hf --seed seed_instructions.json \
    --out self_instruct.jsonl            # -> feed back into Stage 5 build_sft_blend
python 07_teacher/ai_feedback_judge.py --model ./qwen3-ko-sft-hf --in pairs.jsonl \
    --out ko_dpo_pairs2.jsonl --reason-en  # -> feed into Stage 6 DPO

# Non-verifiable CoT data (the gap RLVR can't fill — see GENERAL_COT_GUIDE.md):
python 07_teacher/distill_from_teacher.py --prompts prompts.jsonl --out ko_cot_distilled.jsonl \
    --teacher claude        # 60-80% of CoT data comes from here

# OR (no external teacher available — see SELF_BOOTSTRAP_GUIDE.md, the historical
# STaR recipe that produced the first reasoning models):
python 07_teacher/few_shot_cot_bootstrap.py --model ./qwen3-ko-base-hf \
    --exemplars 07_teacher/exemplars_math_cot.json --problems ko_math_train.jsonl \
    --out-sft ko_cot_bootstrap.jsonl --verifier math --k 16 --rationalize
# same bootstrap for the other verifiable domains, each with its matching exemplars:
#   --verifier code   --exemplars 07_teacher/exemplars_code_cot.json
#   --verifier mcq    --exemplars 07_teacher/exemplars_mcq_cot.json   (e.g. KMMLU/MMLU problems)
#   --verifier logic  --exemplars 07_teacher/exemplars_logic_cot.json
#   --verifier format --exemplars 07_teacher/exemplars_format_cot.json

python 07_teacher/self_consistency.py --model ./qwen3-ko-sft-hf --prompts prompts.jsonl \
    --out ko_cot_consistent.jsonl --k 8
python 07_teacher/critique_revise.py --model ./qwen3-ko-sft-hf --prompts prompts.jsonl \
    --out ko_dpo_critique.jsonl
# then re-run Stage 5/6 with the enlarged data; repeat the loop.

# Stage 8 — RLVR / Korean chain-of-thought (correctness is language-independent)
# 8a. convert a source dataset to {problem, gold} format (see 08_reasoning/converters/README.md)
python 08_reasoning/converters/from_gsm8k.py --split train --out gsm8k_train.jsonl
# 8b. translate problem statements (gold is language-independent, passes through unchanged)
python 08_reasoning/translate_problems.py --in gsm8k_train.jsonl --out ko_math.jsonl
# 8c. rejection-sample verified Korean chains-of-thought, dispatching to the right verifier
python 08_reasoning/rejection_sample.py --model ./qwen3-ko-sft-hf --problems ko_math.jsonl \
    --out-sft ko_math_cot.jsonl --out-dpo ko_math_dpo.jsonl --k 8 --verifier math
python 08_reasoning/verify_math.py --in predictions.jsonl                # sanity-check eval set
python 08_reasoning/grpo_train.py                                        # print the TRL recipe
# ko_math_cot.jsonl folds back into Stage 5 SFT; ko_math_dpo.jsonl into Stage 6.
# Repeat 8a-8c with --verifier {code, mcq, logic, format} and the matching converter.

# Stage 9 — EVAL (run at every milestone checkpoint: pretrain, SFT, DPO, RLVR)
# 1. generate predictions on your frozen eval set (Korean tasks + English-regression slice)
# 2. score fluency, code-switching, verifiable accuracy, and English retention:
python 09_eval/eval_report.py --in predictions.jsonl --verifier math
# watch the English-slice accuracy across checkpoints — a drop = catastrophic forgetting.
```

---

## Using the result as an SFT/DPO lever (the payoff)

After Stage 4 you have a **base** model fluent in Korean that kept its English knowledge — not yet
instruction-following. Bootstrapping order:

1. **Stage 5 SFT** turns it into a Korean instruct model. Seed = your 2M English SFT, machine-translated
   to Korean with the markdown/LaTeX-preserving translator from earlier (`05_sft/translate_en_sft_to_ko.py`
   imports it), plus **passage-grounded native Korean data** generated from the base model via FEW-SHOT
   completion (`05_sft/generate_grounded_ko.py`) — base models can't follow instructions yet, so we
   teach the format by example, ground answers in in-context passages, and filter on faithfulness +
   language-ID. Harvest free grounded pairs from Wikipedia/news structure too (title→article,
   heading→body, headline→summary, cloze).
2. **That instruct model becomes your in-language teacher** — self-instruct / distillation / AI-feedback
   now work *in Korean*, leaning on transferred English reasoning. `07_teacher/self_instruct.py` bootstraps
   more instructions+outputs; `07_teacher/ai_feedback_judge.py` produces preference pairs by judging in
   both orders (position-bias-cancelled), optionally reasoning in English first.
3. **Stage 6 DPO** uses pairs from your own SFT model, labeled with cheap *verifiable* signals
   (correct-language vs code-switched, format-followed, faithful-to-source).
4. **Reasoning (RLVR)** — see `08_reasoning/`: correctness is language-independent — translate problem
   statements via the markdown/LaTeX-preserving translator, reject-sample verified solutions →
   Korean chain-of-thought data even from a modest model. Then GRPO on the same verifier as reward
   for the R1-style RL loop. Works for math, code (unit tests), multiple-choice, format-following —
   any domain with an automatic checker.

Each loop improves the model → improves the data → improves the next model.

---

## Pitfalls checklist
- [ ] Measured fertility before deciding to extend the tokenizer
- [ ] Extended BPE by merging vocab + **merge table** (not `add_tokens`, not SentencePiece)
- [ ] Initialized BOTH input embeddings AND lm_head (Qwen3 unties them)
- [ ] Used subword-average init, not global-mean or random
- [ ] Stage 4b LR ≈ 2e-5 with re-warmup (not pretraining-scale LR)
- [ ] Kept a healthy English replay share; did NOT train pure-Korean
- [ ] Included parallel data for transfer; kept its epochs < ~4
- [ ] Evaluated English AND Korean every checkpoint to catch forgetting early (use `09_eval/eval_report.py`)
- [ ] Protected native 32K context: ran the Stage-4c long-seq retention phase (don't train only at 4096) and probed it with `09_eval/needle_haystack.py` (hi + en, length × depth)
- [ ] In the teacher loop: filtered every batch, kept external signal (real passages / verifiable rewards) each round, mixed in fresh data — to avoid self-training collapse
- [ ] Pinned `transformers>=4.51` and a known-good Megatron-Bridge (or Megatron-LM Qwen converter)
