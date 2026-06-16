# Guide — Designing Grounded Tasks and Exemplars

`generate_grounded_ko.py` supports **eight task types**, each with a matching
`exemplars_<task>.json`. Pick the mix that maximizes coverage of the skills you
want the SFT model to acquire, and run multiple tasks over the SAME corpus to
extract more signal per passage.

## The 8 task types and what each teaches

| Task              | Output shape          | What it teaches the model                          |
|-------------------|-----------------------|----------------------------------------------------|
| `qa`              | one Q + one A         | reading comprehension, focused extraction          |
| `multi_qa`        | 3 Q-A pairs           | density: 3× the supervision per passage            |
| `summary`         | abstractive paragraph | compression, information selection                 |
| `title`           | one headline          | extreme compression, salient-entity selection      |
| `fact_extraction` | bullet list           | structured output, list formatting                 |
| `explain_simply`  | rewritten paragraph   | paraphrase, register shift (formal→simple)         |
| `outline`         | hierarchical numbered | structure, hierarchy, multi-level formatting       |
| `definition`      | one defining sentence | precise, scoped paraphrase from context            |

Run all eight over your Korean corpus and you get one passage producing ~7-10
distinct training examples that exercise different output shapes — far better
than running only `qa` ten times.

## How to write good exemplars

A few-shot generator is only as good as its exemplars. Three rules:

**(1) Diversity across exemplars.** Each exemplar in a file should be on a
different topic, length, and answer pattern — otherwise the model mode-collapses
toward the dominant shape. The shipped files spread across science, history,
geography, culture, and current-affairs deliberately.

**(2) Native Korean quality.** Translationese leaks into outputs. If you can't
hand-write, machine-translate English exemplars and have a native speaker fix
register, idiom, and sentence flow before using them.

**(3) Match the *shape* you want at scale.** For `fact_extraction`, write the
exemplars with bullets you actually want (• or - or numbered, but be consistent).
For `outline`, use the indentation depth you want produced. The model copies the
shape verbatim from the exemplars, so the exemplars ARE the specification.

## Recommended task mix per passage

A pragmatic default for a single Korean corpus:

| Task              | Run rate         | Notes                                             |
|-------------------|------------------|---------------------------------------------------|
| `qa`              | 1× per passage   | the workhorse                                     |
| `multi_qa`        | 1× per passage   | denser than qa; can replace qa to save budget     |
| `summary`         | 1× per passage   | always useful                                     |
| `title`           | 1× per passage   | short, very cheap                                 |
| `fact_extraction` | 1× per passage   | for fact-heavy passages (news, history)           |
| `explain_simply`  | 0.5× per passage | only for technical/abstract passages              |
| `outline`         | 0.3× per passage | only for passages with internal structure         |
| `definition`      | per-term         | needs a `term` field; iterate over key terms      |

Pre-classify passages by content type so you only run `explain_simply` on
technical text and `fact_extraction` on fact-dense text — running every task on
every passage wastes generations on bad matches.

## Filtering knobs per task

`generate_grounded_ko.py` applies a language-ID + faithfulness filter, with
`title` exempted from faithfulness (titles are too short for content-word overlap
to be meaningful). For the strict filter, pipe outputs through
`faithfulness_scorer.py`, which uses the BASE model's PMI and few-shot NLI — that
catches subtle hallucination the lexical check misses, at the cost of more compute.

Recommended per-task post-filters:

- `qa`, `multi_qa`, `summary`, `fact_extraction` → run the strict faithfulness scorer.
- `title` → only language-ID + length sanity (3-15 words).
- `explain_simply` → language-ID only; paraphrase often won't share vocabulary
  with the source, so faithfulness via lexical overlap is too strict here. Use
  PMI from `faithfulness_scorer.py` instead (it rewards paraphrase).
- `outline` → faithfulness via content-word overlap usually works (key entities
  appear in the outline).
- `definition` → faithfulness must pass; a hallucinated definition is a costly error.

## When to write a new task type

Add a new task entry to `TASKS` in `generate_grounded_ko.py` and a matching
exemplar file when you need a *different output shape* not covered above (e.g.
table generation, dialogue from monologue, counterfactual reasoning, ranked
preference list). Add four exemplars, add a chat-template branch to `to_chat()`,
and a parse branch in `parse_completion()`. The selftest pattern in the script
shows how to add a regression test for the new task.
