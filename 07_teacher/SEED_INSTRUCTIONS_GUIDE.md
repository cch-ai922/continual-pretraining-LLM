# Guide — Building `seed_instructions.json` for Self-Instruct

The seed pool is the _only_ signal that shapes what Self-Instruct generates.
The model copies the **distribution** of your seeds — their task types, length,
style, and tone. If your seeds are all short factual Q&A, your 50k generated
examples will be 50k short factual Q&A. Diversity at the seed level is the
single most important lever.

The original Self-Instruct paper used 175 hand-crafted seed tasks. **150-300
high-quality native seeds is the realistic target.** Smaller pools mode-collapse
faster; larger pools dilute the curation effort.

## What makes a good seed

**(1) A complete instruction in itself.** Self-Instruct uses each seed as a
demonstration of "what a task looks like." It should be runnable as-is. The
model will write similar self-contained instructions; if your seeds rely on
unstated context, so will the generated ones.

**(2) Native Korean, no translationese.** Translated seeds bias the model toward
producing translated-feeling text. If you must translate from English, get a
native speaker to rewrite, not just review.

**(3) Specific, not generic.** "이야기를 쓰시오" is too generic and produces a
flood of generic stories. "'진정한 친구'라는 제목의 200자 분량 짧은 이야기를
쓰시오." gives the model a concrete shape to vary on (change topic, length,
title — same shape).

**(4) Varied output shapes.** Mix instructions that produce short, medium, and
long outputs; free-form, list, code, dialogue; analytical and creative.

## Coverage dimensions (mirror the DPO guide)

Cover all of these, with rough target proportions for 200 seeds:

| Category                                    | Count | Examples                                 |
| ------------------------------------------- | ----: | ---------------------------------------- |
| Creative writing (stories, poems, letters)  |   ~30 | "시를 쓰시오…", "편지를 쓰시오…"         |
| Factual / encyclopedic                      |   ~25 | "조선과학자 세 명의 이름과 업적…"        |
| Reasoning / math (with numbers, multi-step) |   ~20 | word problems, arithmetic, algebra       |
| Code (Python/JS, short)                     |   ~15 | "다음의 일을 하는 함수를 작성하시오…"    |
| Explanation (concept, ELI5)                 |   ~20 | "5살 아이에게 {X}를 설명하시오"          |
| Advice / planning                           |   ~15 | "{X}를 위한 단계별계획을 제안하시오"     |
| Comparison / contrast                       |   ~10 | "{X}와 {Y}의 차이를 설명하시오"          |
| Summarization / paraphrase                  |   ~10 | "이 문단을 쉬운 말로 다시 쓰시오"        |
| Lists / structured                          |   ~15 | "{X}의 우점 다섯가지를 목록으로 쓰시오"  |
| Translation (EN↔KO both directions)         |   ~10 | "이 문장을 조선어로 번역하시오"          |
| Dialogue / roleplay                         |   ~10 | "{교원/학생, 의사/환자}의 대화를 쓰시오" |
| Cultural / Korea-specific                   |   ~20 | 명절, 속담, 력사유적                     |

The shipped `seed_instructions_sample.json` has 40 covering these proportions —
use it as a starting shape reference and scale up.

## How to scale to 200-300

**Hand-write 50** carefully, one per task type from the table above. These are
the highest-quality "anchors." Vary topic and length within each row.

**Crowd-source / commission 100-150** from native Korean speakers using your
hand-written 50 as a written brief. Pay for diversity, not volume — explicitly
ask for instructions that look DIFFERENT from the brief examples.

**Mine your existing data for 50-100.** The user turns in your translated SFT
(`ko_sft.jsonl`) are valid instructions — filter for ones that are self-contained
and natively-phrased after translation. Same for any human-written prompt logs
you have access to.

**Adapt the canonical Self-Instruct seed list (175 English tasks).** Translate
the structural diversity (not the literal content) into native Korean. Use it as
a checklist: did you cover "rewrite", "expand", "classify", "extract",
"reformulate", "explain to age N"? If a category is missing, write a few.

## Validation before committing

After assembling the pool, run these checks:

```bash
# 1. exact-duplicate check (should be 0)
python -c "import json; s=json.load(open('seed_instructions.json')); print(len(s) - len(set(s)))"

# 2. token-jaccard near-duplicate scan (use self_instruct's helper)
python -c "
import json, sys
sys.path.insert(0,'07_teacher')
from self_instruct import too_similar
s = json.load(open('seed_instructions.json'))
dups = [(i,j) for i in range(len(s)) for j in range(i+1,len(s)) if too_similar(s[i],s[j], 0.6)]
print(f'near-dup pairs at sim>=0.6: {len(dups)}')
for i,j in dups[:5]: print('  ', s[i][:60], '||', s[j][:60])
"

# 3. language-ID: every seed should be majority Korean
python -c "
import json, unicodedata
s = json.load(open('seed_instructions.json'))
def hi(x):
    L=[c for c in x if c.isalpha()]
    return sum(1 for c in L if 'HANGUL' in unicodedata.name(c,''))/max(len(L),1)
bad = [t for t in s if hi(t) < 0.7]
print(f'low-Korean seeds: {len(bad)}')
for b in bad[:3]: print('  ', b[:80])
"

# 4. length distribution (avoid all-short or all-long pools)
python -c "
import json, statistics
s = json.load(open('seed_instructions.json'))
lens = [len(x.split()) for x in s]
print(f'words: min={min(lens)} median={statistics.median(lens)} max={max(lens)}')
"
```

## What happens after Self-Instruct runs

`self_instruct.py` adds every survivor to the pool and uses it as a seed for
later rounds. So **the seed pool grows during the run** with model-generated
instructions. Two safeguards:

- **Keep the original hand-written seeds tagged** (e.g. as the first N entries)
  so you can re-seed from clean ones each round if drift becomes a problem.
- **Inspect 100 randomly-sampled generated instructions after each round.** If
  you see mode collapse (e.g. 30 of 100 are "이야기를 쓰시오" variants), tighten
  the similarity threshold (raise `--sim-thresh` from 0.7 to 0.6 or 0.55) or
  restart from the original seeds with different sampling temperature.

## Pitfalls

- **Translation-heavy seed pool** → outputs feel un-Korean. Fix: rewrite, don't translate.
- **All seeds same length** → outputs collapse to that length. Fix: span short/medium/long.
- **No code/math seeds** → those skills decay from your SFT model in the loop.
- **No Korean cultural content** → outputs become culturally-English. Fix: ≥20% Korea-specific seeds.
- **Skipping validation** → near-duplicate seeds waste the diversity budget. Always run the jaccard scan.
