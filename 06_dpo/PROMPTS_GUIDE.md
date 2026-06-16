# Guide — Building `prompts.jsonl` for DPO

`build_preference_data.py` reads JSONL where each row is:

```json
{
  "prompt": "...",
  "format": "json|list|null",
  "source": "optional grounding text"
}
```

For each prompt it samples two responses from your SFT model and labels them by
the verifiable signals (language correctness, format adherence, faithfulness,
non-degeneracy). The **quality and diversity of the prompts** directly determines
what your DPO-trained model gets better at — narrow prompts → narrow improvement.

## What to optimize for

DPO has highest leverage on prompts where the SFT model **sometimes fails** —
prompts that produce _some_ good outputs and _some_ bad ones at temperature ≥ 0.7.
A prompt the model always handles perfectly produces ties and yields no signal;
a prompt it always fails on produces two bad responses with no clear winner.
Aim for the middle: borderline-difficulty prompts where the verifiable signals
will sort the samples cleanly.

## Coverage dimensions

Build the file so the prompt set spans **all** of these dimensions, not just one.
Mode collapse on any axis costs you.

### 1. Task type (the most important axis)

- **Factual Q&A** — Korean and general knowledge ("조선민주주의공화국의 수도는?")
- **Reasoning / math** — word problems, simple algebra, multi-step arithmetic
- **Code** — Python/JavaScript, explain/generate/debug, kept short
- **Creative writing** — stories, poems, dialogues, letters
- **Summarization** — with `source` field so faithfulness can score
- **Explanation** — concepts, simple-language rewrites
- **Advice** — interpersonal, professional, lifestyle (be careful with anything medical/legal)
- **Translation** — EN→KO and KO→EN
- **Structured output** — JSON, list, table; ALWAYS set the `format` field
- **Comparison** — A vs B contrasts
- **Conversation/roleplay** — chat turn, customer-service, teacher-student

### 2. Cultural register

A bilingual model often falls back to English-centric content. Force native
Korean context in ~40% of prompts: Korean history, geography, festivals, cinema,
food, sports (e.g. baseball/football), domestic politics, Korean companies, Korean literature.

### 3. Difficulty

- **Easy** (~30%): single-step, single-fact
- **Medium** (~50%): 2–3 reasoning steps, multi-sentence answer
- **Hard** (~20%): multi-step reasoning, structured output, long-form

### 4. Length expected in response

- **Short** (≤50 words): factoid, title, definition
- **Medium** (50–200 words): explanations, summaries
- **Long** (200+ words): stories, detailed analyses

### 5. With/without `source`

~25% should include a `source` passage so the faithfulness signal can fire.
These produce the strongest preference signal because hallucination is detectable.

### 6. With/without `format`

~15% should set `format: "json"` or `format: "list"` so format-adherence becomes a
verifiable signal. Format failures are the easiest DPO signal to obtain.

## How to scale to 50k–500k prompts

**Translate-and-rewrite from English.** Take an existing English prompt set
(LMSYS-Chat-1M user turns, OpenAssistant prompts, ShareGPT prompts) and translate
through the markdown/LaTeX-preserving translator (`05_sft/translate_en_sft_to_ko.py`
imports it). Filter to coherent Korean (use the language-ID check). This is the
fastest way to get to volume.

**Mine your own SFT corpus.** Every Stage-5 SFT example has a user turn that is
already a prompt. Extract them: `jq '.messages[0].content' ko_sft_blend.jsonl`.

**Self-instruct overlap.** The instructions from `07_teacher/self_instruct.py`
are also valid DPO prompts; the same pool serves both stages.

**Templated generation.** Pick a list of 200 Korean nouns/entities (people,
places, concepts) and instantiate templates like "{X}에 대해 설명하시오",
"{X}와 {Y}의 차이는 무엇입니까?", "{X}를 요약하시오". Cheap, diverse, and you
control the distribution exactly. Run dedup afterward.

**Wikipedia / news topic seeds.** Wikipedia article titles and news headlines
make excellent prompts ("{headline}에 대한 분석을 작성하시오"). Free, native, current.

**For `source`-bearing prompts:** pair a real Korean passage with one of:
summary, fact-extraction, grounded-QA, faithful-paraphrase, title. These
overlap with the grounded-generation tasks in Stage 5, so you can reuse the
same passage pool.

## Filters to apply BEFORE training

Before passing through `build_preference_data.py`, dedupe and clean the file:

```bash
# de-duplicate by exact prompt match
jq -c '.prompt' prompts.jsonl | sort -u | wc -l   # check unique count

# drop overly-short prompts (low signal)
jq -c 'select(.prompt | length >= 15)' prompts.jsonl > prompts_clean.jsonl

# language-ID: drop prompts with too little Korean
python -c "
import json, unicodedata, sys
def hi(s):
    L=[c for c in s if c.isalpha()]
    return sum(1 for c in L if 'HANGUL' in unicodedata.name(c,''))/max(len(L),1)
for line in sys.stdin:
    o = json.loads(line)
    if hi(o['prompt']) >= 0.5: print(line, end='')
" < prompts_clean.jsonl > prompts_final.jsonl
```

## Pitfalls

- **Length skew** — if all prompts are short Q&A, DPO won't help long-form. Force the long-form mix in.
- **Topic skew toward English-centric content** — translated prompts inherit US/UK cultural references. Manually re-weight toward Korean content.
- **No format prompts** — without `format`-tagged prompts, the format signal in `build_preference_data.py` is never exercised.
- **No `source`-bearing prompts** — without these the faithfulness signal is dead weight; you're throwing away one of the strongest verifiable rewards you have.
- **Generating prompts WITH the SFT model** — this is fine for volume, but the prompts inherit the model's blindspots. Always mix in human/translated sources.

## A reasonable target distribution

For a 100k-prompt set:

| Category                 | Count | `format`  | `source` |
| ------------------------ | ----: | --------- | -------- |
| Factual Q&A (Korean)     |   20k | none      | none     |
| Factual Q&A (general)    |   10k | none      | none     |
| Reasoning / math         |   12k | none      | none     |
| Code (short)             |    8k | none      | none     |
| Creative writing         |   12k | none      | none     |
| Summarization (grounded) |   12k | none      | yes      |
| Grounded Q&A             |    8k | none      | yes      |
| Explanation / advice     |   10k | none      | none     |
| Structured output        |    6k | json/list | optional |
| Translation              |    2k | none      | none     |

`prompts_sample.jsonl` in this directory has 25 prompts hitting all of these
categories — use it as a small starting set and as a shape reference when you
scale up via the methods above.
