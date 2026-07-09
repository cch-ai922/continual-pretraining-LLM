---
title: "What Must Be Trained and What Can Be Prompted: A Per-Capability Study of Tutor Redirect Behavior"
author: "Miles Yung — Independent Research (milesyung2026@gmail.com)"
abstract: |
  Deploying a small language model as an English tutor forces a question a
  flat "good-dialogue" corpus never does: *which* tutor behaviors can be
  elicited by an explicit system prompt, and which must be demonstrated
  through fine-tuning? We answer this per-capability under a
  **matched-prompt** protocol --- the identical fully-specified prompt
  given to a fine-tuned 0.8B student and to prompt-only baselines up to a
  9B teacher. A sharp boundary emerges (established in Qwen, then
  replicated in a trained Llama student). Behaviors one clause elicits ---
  locale fidelity, role-swap deflection, topic re-anchoring --- reach
  prompt-only parity. Behaviors the prompt *describes but cannot install*
  do not: on persistence (firing a session-end sentinel on the third
  same-axis violation) the 9B teacher reaches $\leq$ 0.06
  recall zero/few-shot and only 0.63 with native chain-of-thought (below
  the student's 0.83); on pedagogical withholding, prompt-only models
  manage 0.09--0.45 versus the student's 0.61. The boundary *replicates in
  a second trained family*: a Llama-3.2-1B student, against its own
  untrained base under the same prompt, reaches 0.91 persistence recall
  and 0.50 withholding versus 0.25 and 0.11 prompt-only --- isolating
  training from scale. All experiments run on one RTX 3060 12GB GPU.
---



# 1. Introduction

**Problem.** We ask, for each behavior a deployed English tutor needs,
whether a sufficiently explicit system prompt is enough or the behavior must
be demonstrated in fine-tuning. A deployed tutor is governed by a long
system prompt that already *states* most of what it should do — stay in
character, ground culture in the learner's locale, refuse role swaps, give
one short example rather than a grammar lecture, escalate to a session-end
signal under sustained abuse. If stating a behavior were sufficient, the
data-side problem would be trivial.

**Why it matters.** It is not: the behaviors split cleanly into ones a
prompt clause elicits and ones it only describes, and knowing which is which
is exactly the decision a practitioner faces when building a task-specific
model on consumer hardware — spend the data budget only where a prompt will
not do. The tutor setting makes the question answerable because each target
behavior corresponds to a clause already present in a realistic deployment
prompt (§3): the language of instruction, lesson topic, role structure,
persona, pedagogical contract, locale frame, and general appropriateness.

**Gap.** The prompting-versus-fine-tuning trade-off has been studied at the
level of *general* alignment — that a small demonstration set installs broad
instruction-following [@zhou2023lima; @ouyang2022instructgpt], and that
in-context demonstrations often convey format more than new capability
[@min2022rethinking] — but not resolved *per behavior* for a deployed task
model. Synthetic-instruction pipelines and tutor-LLM datasets (§2) target
general instruction-following with a uniform recipe; none asks, per
behavior, whether the deployment prompt already elicits what the data
teaches.

**Approach and contributions.** We supply the same fully-specified
deployment prompt at evaluation to a fine-tuned 0.8B student and to a ladder
of prompt-only baselines (0.8B base, 0.8B instruct, 4B instruct, and the 9B
teacher that generated the training data; Figure 2). Because
the instruction is present for every condition, a prompt-only failure cannot
be blamed on under-specification. Our contributions:

1. **A per-capability map of the train-versus-prompt boundary**
   (Figure 3): promptable when a single clause both
   *describes* and *elicits* the behavior (locale fidelity, role-swap and
   topic re-anchoring), not promptable when it requires cross-turn
   state-tracking (persistence) or the suppression of a strong competing
   prior (pedagogical withholding).
2. **An evidenced negative result about the conventional metric**: per-axis
   redirect F1 is a context-blind *type* classifier that ties a 0.8B student
   with the 9B teacher; we replace it with matched per-capability
   instruments (§5).
3. **Reusable apparatus**: a locale-aware, yield-aware generation pipeline
   (§3) and a *partially validated* trigger-position decorrelation
   construction for the persistence sentinel (§5).

All experiments run on one RTX 3060 12GB GPU — the regime where the boundary
matters most, since a large general-purpose model is not deployable there. We
evaluate one family (Qwen; a genuine limitation, §6) and one locale (which
does not limit the persistence/withholding boundary, both structural
behaviors independent of the locale backdrop).



# 2. Related Work

**Synthetic instruction data.** Teacher-to-student bootstrapping was
popularised by Self-Instruct [@wang2023selfinstruct] and Evol-Instruct /
WizardLM [@xu2023wizardlm]; UltraChat [@ding2023ultrachat] extends to
multi-turn dialogue, and Tulu-3 [@lambert2024tulu3] composes datasets from
sub-pools. These target *general* instruction-following with a uniform
recipe and do not ask which taught behaviors a prompt already elicits.

**LLM-as-judge filtering.** Prometheus [@kim2024prometheus], JudgeLM
[@zhu2023judgelm], and PandaLM [@wang2023pandalm] propose dedicated judges;
a strand documents position, length, and self-preference bias
[@saito2023verbosity; @panickssery2024selfpreference], mitigated by
multi-judge consensus. Our pairwise eval (§5) follows this with a
cross-family constraint so no judge shares the teacher's lineage; we add a
*negative* result about a judged metric (context-blind redirect F1).

**Tutor and educational LLMs.** EduChat [@dan2023educhat] and MathDial
[@macina2023mathdial] target educational dialogue; CEFR-aligned corpora
(EFCAMDAT [@geertzen2014efcamdat], TLE [@berzak2016tle]) are
*learner-produced*, not tutor-side. None asks, per behavior, whether the
deployment prompt already suffices. MathDial's scaffolding moves are the
closest analogue to our pedagogical-withholding axis.

**Safety, persona, and shortcut learning.** Adversarial safety data
[@parrish2022bbq; @mazeika2024harmbench; @zou2023advbench] and persona
consistency [@zhang2018personas; @deshpande2023toxicity] mostly treat
redirection as *single-turn*. Our three-strike persistence is multi-turn and
carries a *shortcut-learning* [@geirhos2020shortcut; @mccoy2019hans] risk: a
positionally-regular sentinel invites a position-as-trigger shortcut, which
our decorrelation construction (§3) targets.

**Positioning.** The prompting-vs-fine-tuning question has been examined for
*general* alignment [@zhou2023lima; @ouyang2022instructgpt; @min2022rethinking];
we localize it to the *per-behavior* level for a deployed task model, under a
matched-prompt protocol giving the same prompt to trained and prompt-only
models alike.



# 3. Method

![**The locale-aware, yield-aware generation pipeline.** A single locally-served 9B teacher drives twelve SFT streams; a six-filter cascade and a yield-aware top-up loop shape the corpus, which trains a 0.8B QLoRA student scored on six frozen held-out sets — all on one RTX 3060 12GB.](paper_workshop/figures/fig_pipeline.png)

**Pipeline.** A single locally-served 9B teacher generates all training
data from a pool of CEFR-stratified scenario seeds through twelve parallel
SFT streams, a six-filter cascade, and a yield-aware top-up loop
(Figure 1). The teacher receives a prompt of exactly the
shape the student sees at deployment, so there is no distribution shift
between demonstration and inference. Full pipeline mechanics (top-up
equations, filter cascade, hash-deterministic split) are in the appendix and
the released code.

**An invariant-based taxonomy.** We read the redirect axes off the
deployment prompt rather than intuiting them. A tutor is the keeper of a set
of **interaction invariants**, each corresponding to one commitment the
prompt makes; a learner *violation* breaks exactly one, and the correct
redirect is the **minimal repair** that restores it
(Table 1). This makes the axis list auditable
(one clause ↔ one axis) and converts "a generic redirect stream is
insufficient" into a per-axis falsifiable prediction: specialized data
should help exactly on axes whose repair is not already supplied by
assistant priors or a prompt clause (§5).

| **Invariant / violation** | **Minimal repair** |
| :-------------------------------- | :----------------------------------------- |
| Language (code-switch to L1) | Acknowledge, steer back to target language |
| Topic (off-topic drift) | Re-anchor to the subject |
| Role ("you be the learner") | Decline, reassert tutoring structure |
| Persona ("are you a chatbot?") | Reassert the frame, continue in persona |
| Pedagogy ("just give the answer") | Scaffold rather than supply the answer |
| Locale (out-of-locale reference) | Brief in-locale redirect |
| Appropriateness (politics/etc.) | Generic safe redirect (catch-all) |

: **The seven single-shot redirect axes** as
invariant/violation/minimal-repair triples, read off the deployment
prompt. {#tab:invariant-triples}

The pipeline realises this through twelve streams: *normal* scaffolded
dialogue (1), *single-shot redirects* (7, one per axis), and *persistent
3-strike* streams (4, for the axes where repeated violation warrants
escalation to a session-end sentinel).

**Deployment prompt.** Every axis appears as an explicit `[guidelines]`
clause, and a `[persistence]` block specifies the full three-strike protocol:
first warm redirect, second shorter redirect, third a warm sentence plus the
exact literal sentinel `[SESSION_END: <axis>]`. This is what makes the
boundary interpretable — when a prompt-only model fails on persistence or
withholding (§5), it fails *despite* the behavior being spelled out.

**Trigger-position decorrelation.** A fixed-turn sentinel makes turn
position a perfect proxy for "third strike," inviting a positional shortcut.
We resample the sentinel turn over a parity-constrained set {5, 7, 9, 11}
(hash-deterministic per seed; every variant holds the trigger at exactly
three strikes, varying only lead-in scaffolding). We report this as
*partially validated* (§5).



# 4. Experimental Setup

**Models and recipe.** Student base `Qwen3.5-0.8B-Base`; teacher
`Qwen3.5-9B-UD-Q4_K_XL` (4-bit, llama.cpp), which also serves as the
strongest prompt-only baseline (B4). All experiments on one RTX 3060 12GB.
Every trained condition uses the *same* recipe — QLoRA [@dettmers2023qlora]
(NF4 4-bit) with LoRA [@hu2022lora] rank 16 / alpha 32, 2 epochs, no DPO —
so each ablation is single-variable (hyperparameters in the appendix).

**Conditions.** **A1** — the student, full 12-stream mix. **A3** — A1 minus
the six specialized single-shot redirect streams (the *generic-SFT*
baseline). **A5** — A1 with persistence rebuilt fixed-turn-7 (the
decorrelation contrast). Prompt-only: **B1** 0.8B base, **B2** 0.8B
instruct, **B3** 4B instruct, **B4** 9B teacher. All see the identical
deployment prompt (Figure 2).

**Held-out sets** (frozen, `locale=china`): Tutor-Scenario (224),
Redirect-Probe (143), and four persistence probes isolating recall
(Persistent-Probe, 159 positives) from two false-positive channels
(Persistent-FP-Probe benign negatives at trained positions;
Persistent-Premature-Probe under-threshold negatives) and off-grid firing
(Persistent-OffPosition-Probe).

**Metrics.** Persistence and locale leakage are **mechanical** (judge-free):
sentinel-firing recall against literal `[SESSION_END: <axis>]` strings, and
Western-default leakage via a word-boundary gazetteer. Withholding is a
binary withheld/answered rate under two judges (Llama-3.1-8B, Gemma-2-9B).
Repair *quality* on the promptable axes is a pairwise preference under a
cross-family three-judge ensemble (Prometheus / Llama-3.1 / Gemma-2, all
distinct from the Qwen teacher to eliminate self-preference bias). The
load-bearing A1/A3 conditions are reported over **three seeds** (42/123/7);
others single-seed.



# 5. Results: The Train-versus-Prompt Boundary

Every condition is evaluated under the **same** fully-specified deployment
prompt (§3), so a prompt-only failure isolates *promptability* rather than
under-specification (Figure 2). The behaviors separate sharply
(Figure 3, Table 2).

![**Matched-prompt protocol.** The identical deployment prompt is given to a fine-tuned 0.8B student, a generic-SFT ablation, and prompt-only baselines up to the 9B teacher; each is scored by the instrument matched to the capability.](paper_workshop/figures/fig_design.png)

![**The boundary.** *Top*: behaviors one clause elicits — prompt-only reaches parity. *Bottom*: behaviors the prompt describes but cannot install — prompt-only falls far short of the trained 0.8B student. Values from Table 2.](paper_workshop/figures/fig_boundary.png)

| **Capability (metric)** | **Prompt-only**                        | **Trained** |
| :------------------- | :----------------------------------- | :--------- |
| Persistence (recall)    | $\leq$ 0.06 / A3 0.000 | **0.83**    |
| Withholding (rate)      | 0.09--0.45                             | **0.61**    |
| Locale (1$-$leakage)    | 0.987                                  | 0.987       |
| Role-swap (type F1)     | 0.71--0.86                             | 0.75--0.86  |
| Topic (type F1)         | saturated                              | saturated   |

: **The boundary.** *Top*: not-promptable (the 9B teacher and,
for persistence, the no-data ablation A3 fail despite the explicit
instruction). *Bottom*: promptable (prompt-only parity). Persistence
prompt-only is the 9B teacher's zero/few-shot recall; native CoT reaches
0.63 (§5.1). Withholding prompt-only spans B1--B4 (9B teacher 0.45).
{#tab:boundary}

## 5.1 Not promptable I — multi-turn persistence

On the third same-axis violation the tutor must emit the exact literal
sentinel. Under the fully-specified prompt, no prompt-only model fires it
reliably: the 9B teacher reaches $\leq 0.06$ recall zero/few-shot, and the
no-data ablation A3 fires on **0.000** of positives, while every model
trained on the persistent streams fires (A1 recall **0.83**, 3-seed mean
$0.85 \pm 0.04$).

**The gap is not an artifact of zero-shot prompting.** We ran a prompting
ladder on the strongest prompt-only model (the 9B teacher; few-shot exemplars
span positions {5,7,9,11} so they cannot teach a fixed turn).
Table 3 and Figure 4 (left) show few-shot and
an output counting-scaffold do *not* help ($\leq 0.06$); only Qwen3.5's
*native* chain-of-thought moves recall, to 0.63 — still below the trained
0.83, and at 1.6–3.2k reasoning tokens/turn with frequent delivery failures
(the fire decision is reached inside `<think>` but omitted from the visible
answer). Persistence is thus a *relocated* boundary: it resists zero/few-shot
outright and is only partially recovered by native CoT, at an accuracy,
reliability, and inference-cost deficit SFT removes.

| **9B teacher prompting condition** | **recall** |
| :-------------------------------------------------- | :-------- |
| zero-shot instruction              | 0.025      |
| \+ few-shot exemplars              | 0.013      |
| \+ CoT output scaffold (no-think)  | 0.057      |
| \+ native chain-of-thought         | **0.63**   |
| A1 trained 0.8B (reference)        | **0.83**   |

: **Persistence resists prompting up to, but not including,
native CoT --- and even that stays below the trained student.** Recall
on Persistent-Probe positives (n=159). {#tab:ladder}

**Decorrelation, partially validated.** In an isolated matched contrast (A1
4-variant vs A5 fixed-turn-7, 1-epoch budget), decorrelation *removes the
positional recall bias* — the fixed-turn design recalls best at its trained
turn 7 and degrades off-position (overall 0.56), while the 4-variant fires
uniformly across positions (0.82) — but does *not* reduce premature firing
(0.208 vs 0.119). At 0.8B the premature over-firing tracks accumulated
violation count and conversation depth (peaking at turn 9, not the trained
7), so it is threshold-laxity, not a positional shortcut for decorrelation to
suppress.

## 5.2 Not promptable II — pedagogical withholding

The tutor must *withhold* the requested answer and scaffold instead, an
instruction the prompt states explicitly. We score 63 held-out probes under
two judges (Table in Figure 4, right). The trained student
withholds at **0.61**; the no-specialized-data ablation A3 collapses to 0.12,
near the untrained-base rate — clean evidence the behavior is *installed by
demonstration* (A1-vs-A3 clears significance under each judge, $z=5.9$ /
$5.6$, $p \ll 0.001$). Every prompt-only model, including the 9B teacher (0.45),
withholds less than the trained 0.8B student despite the identical
instruction: a model an order of magnitude larger, told plainly to scaffold,
complies on under half of probes. (The student-beats-teacher *comparison* is
directional only at n=63; the boundary rests on the teacher's absolute
non-compliance and the A3 ablation.) Scaffolding requires suppressing the
model's strong helpfulness prior — a clause can describe that suppression but
not produce it.

![**The two not-promptable behaviors.** *Left*: on persistence, the 9B teacher stays at $\le$0.06 through few-shot and an output CoT scaffold; only native CoT moves it, to 0.63 — still below the trained 0.83 (dashed). *Right*: on withholding, every prompt-only condition (incl. the 9B teacher, 0.45) falls below the trained 0.61, and the no-data ablation A3 collapses to baseline.](paper_workshop/figures/fig_failure.png)

## 5.3 Promptable axes — type is prompted, quality is refined

Redirect-axis F1 from a context-blind judge is **uninformative**: it is a
*type* classifier that floors on axes with no context-free surface form
(locale, language, topic) and ceilings on axes every model satisfies
(persona, role-swap), tying A1, B2, and the 9B teacher at 0.409. We therefore
score *quality* on the promptable axes with a pairwise preference (A1 vs A3,
both SFT-only, differing only in the specialized streams). Every axis favors
A1; the largest effect is **role-swap (win-rate 0.87)** — on an axis F1
called saturated parity, so the F1 ceiling concealed a real, large
specialized-data effect. A generic-redirect negative control (both conditions
train on it) sits at 0.55, so the wins are axis-specific, not a global "A1 is
better" artifact. **Locale fidelity is prompt-driven**: Western-default
leakage is statistically indistinguishable across trained and prompt-only
models (A1 1.34%, A3 0.89%, B1 1.34%), and ablating the locale stream does
not increase leakage — it is carried by one prompt clause.



# 6. Discussion and Conclusion

Under a matched-prompt protocol, tutor redirect behaviors separate along an
interpretable line: **promptable** when a single clause both *describes* and
*elicits* the behavior, and **not promptable** when it requires cross-turn
state-tracking (persistence) or the suppression of a strong competing prior
(withholding) — things a clause can name but not produce. The boundary tells
a deployer of a small model which behaviors a prompt gives for free (locale,
role-swap, topic) and which require the pipeline (persistence, withholding).
Reaching it cleanly required retiring the conventional context-blind
redirect-axis F1 — a *type* classifier that ties a 0.8B student with a 9B
teacher — for matched per-capability instruments, which recover the largest
specialized-data effect in the study (role-swap quality, win-rate 0.87) on an
axis F1 called saturated parity.

**Cross-family replication.** The boundary is not a Qwen artifact: it
replicates in a second *trained* family. A Llama-3.2-1B student, evaluated
against its *own* untrained base under the identical prompt, lifts both
not-promptable behaviors far above prompt-only — persistence recall 0.25$\rightarrow$0.91,
withholding 0.11$\rightarrow$0.50 (two judges) — isolating training from scale, since
student and control share one base. A larger Llama-3.1-8B prompt-only probe
stays low even with chain-of-thought (persistence 0.27$\rightarrow$0.55; withholding
0.22–0.32). The boundary's *direction* is robust across families; the
*magnitude* is family-dependent (Llama trained withholding 0.50 vs Qwen 0.61,
and Llama attains more prompt-only persistence than the Qwen teacher). One
asymmetry: the Llama student trains from an instruct checkpoint, since
Llama-3.2-1B-*Base* could not learn the rare turn-end token under LoRA-SFT.

**Threats to validity.** *Single locale*
does not threaten the central boundary: persistence and withholding are
structural behaviors independent of the locale backdrop. *Judged-metric
power*: the trained-vs-teacher withholding gap is directional at n=63; the
strong claim is the teacher's absolute sub-50% compliance. *Decorrelation* is
partially validated (fixes positional recall, not premature firing).

**Future work**, in priority order: (i) **broader cross-family replication**
(a third family, a base-checkpoint student, multi-locale) — the trained Llama
student above establishes the boundary in a second family; breadth remains;
(ii) **larger-scale decorrelation**, where the positional route is
cheaper relative to the semantic one; (iii) a **better-powered pedagogy
teacher comparison**; and (iv) transfer of the matched-prompt methodology to
other rare, semantically-triggered markers (refusal-token and tool-call
emission), where the same "describable but not promptable" question applies.

**Conclusion.** Which tutor behaviors must be trained and which can be
prompted is answerable per capability, and the answer is sharp: a prompt
clause installs a behavior only when it both describes *and* elicits it.
Persistence and pedagogical withholding are described by the prompt but
installed only by demonstration — the central finding, established on a single
consumer GPU where the distinction matters most.

## Ethics and data statement

All training and evaluation data are model-generated (distilled from a
locally-served teacher) and contain no personal data or PII; no human
subjects were involved. The intended use is research on data curation and the
train-versus-prompt boundary for small task-specific models.



## Code and Data Availability

Code, training/evaluation scripts, configuration, and the synthetic training/evaluation datasets are available at <https://github.com/cch-ai922/tutor-train>. The released datasets are model-generated (teacher-distilled) and contain no personal data. A full-length version of this paper reports the per-capability statistics and the complete confound analysis.




# A. Reproducibility and Hyperparameters

**Training.** QLoRA [@dettmers2023qlora] (NF4 4-bit, double-quant,
`bfloat16`) with LoRA [@hu2022lora] rank 16, alpha 32, dropout 0.05 over the
seven linear projections of the language tower; 2 epochs, peak LR
$2\times10^{-4}$ cosine, effective batch 8, max sequence length 1792,
`paged_adamw_8bit`, SDPA attention. No DPO enters any condition. Teacher
served via llama.cpp (context 32 768, 80 GPU layers); teacher and trainer are
swapped since they cannot co-reside on 12 GB.

**Persistent-variant codomain.** Sentinel positions {5,7,9,11} are forced:
dialogues open on a learner turn (only odd positions valid), with $\geq 5$ to
fit three strikes and two probes and $\leq 11$ to stay under the 1792-token
cap. Variant assignment `int(sha256(seed_id)[:8]) % 4` is seeded (not random)
so re-generation is stable and the train/eval split stays coherent.

**Pipeline filters.** Six filters run cheap-to-expensive: five mechanical
(script, banned terms, schema, naturalness) then one LLM-judge (`locale_judge`).
An audit of the `locale_judge` found ~85% false positives on its rejections
(common English sentence-initial words, locally-canonical landmarks),
remediable with static allowlists (global pass rate 70.1% $\to$ 88.4%) — a
reusable caution for any capitalization-based entity filter.

**Artifacts.** One Python codebase; generation driven by
`config/generation.yaml`, training by per-condition YAMLs, the six held-out
sets frozen by `scripts/build_eval_sets.py`. The full-length version of this
paper reports the per-capability statistics, the F1-bimodality negative
result, and the complete confound analysis.

**Code and Data Availability.** All code, training configurations, seeds, frozen
evaluation sets, judge prompts, synthetic datasets, and per-condition score
outputs are released at <https://github.com/cch-ai922/tutor-train>, with a
`reproducibility/` guide mapping each result to the script and config that
produce it. Trained adapters are deltas over the public base models
(Qwen3.5-0.8B-Base; Llama-3.2-1B-Instruct) and are available on request.
