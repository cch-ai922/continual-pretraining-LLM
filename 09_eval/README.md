# Stage 9 — Evaluation

The piece the original pipeline was missing. The project's goal is **Korean fluency
without English forgetting**, and the pitfalls checklist says to _"evaluate English
AND Korean every checkpoint"_ — but there was no tool to do it. `eval_report.py` is
that tool.

## What it measures

- **Korean fluency proxy** — mean Hangul fraction on the Korean slice.
- **Code-switching rate** — % of Korean-expected responses leaking Latin script.
- **Verifiable accuracy** — dispatches to `../08_reasoning/verify_<name>.py`
  (math, mcq, code, logic, format) for rows that carry a `gold` answer.
- **English regression** — the same metrics on rows tagged `lang:"en"`. **This is
  how you see forgetting**; watch the English accuracy across checkpoints.

## Workflow

1. Build a **frozen** eval set (see the data-labour guide, G4) covering Korean task
   types + a held-out **English-regression** slice.
2. Generate predictions with your serving stack (vLLM sketch is in the script
   docstring). Each prediction row:
   ```json
   {"prediction": "<model text>", "lang": "ko", "gold": <optional>, "n_options": 4, "verifier": "math"}
   ```
3. Score:
   ```bash
   python eval_report.py --in predictions.jsonl --verifier math
   python eval_report.py --in predictions.jsonl            # fluency/code-switch only
   python eval_report.py --in predictions.jsonl --out per_row.jsonl
   ```
4. Run it **every milestone checkpoint** (Stage 4 pretrain, Stage 5 SFT, Stage 6
   DPO, Stage 8 RLVR). Log the per-slice numbers over time.

## Pure-logic self-test (no model/GPU needed)

```bash
python eval_report.py --selftest
```

## Calibration

The code-switch thresholds (`--min-korean 0.6`, `--max-latin 0.15`) are proxies.
Calibrate them on ~150 hand-labeled responses (data-labour guide G1/G5); proper
nouns and accepted loanwords (콤퓨터, 뻐스) should not trip the flag.

## Long-context probe — `needle_haystack.py`

`eval_report.py` does not measure context length. Qwen3-8B is native 32K, but the
Stage-4b continual run at seq 4096 never trains positions 4096..32768, so long
context can silently erode. `needle_haystack.py` buries a unique code (the needle)
at a known depth inside a haystack of a target token length and checks retrieval —
swept over length × depth, in **both Korean and English**.

```bash
# build probe sets (use your real tokenizer so token lengths are accurate)
python needle_haystack.py --build --lang hi --tokenizer ./qwen3-ko-base-hf \
    --lengths 4000,8000,16000,32000 --depths 0,0.25,0.5,0.75,1.0 --out probes_ko.jsonl
python needle_haystack.py --build --lang en --tokenizer ./qwen3-ko-base-hf --out probes_en.jsonl
# generate predictions (vLLM sketch in the script docstring), then score:
python needle_haystack.py --score --in predictions.jsonl
python needle_haystack.py --selftest
```

Run it at every milestone checkpoint, especially **before and after Stage 4c**
(the long-context retention phase) so you can prove the phase worked. A cell that
collapses to ~0% means context is broken at that length/position. For a realistic
haystack, pass `--filler-file` with your own long Korean/English text.
