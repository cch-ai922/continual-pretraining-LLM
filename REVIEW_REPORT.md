# Korean-Qwen3-8B — Port & Review Report

This project is a full **Hindi → Korean port** of the original Qwen3-8B adaptation
pipeline. Every module was re-localized to Korean (Hangul) and re-verified; the
self-tests are the correctness spec.

## Verification status

- **27 `--selftest` suites PASS** + 1 expected skip (`02_embeddings/init_replaced_embeddings.py`
  needs `torch`, absent here — intrinsic, not a defect).
- **`07_teacher/test_cot_exemplars.py`: 25/25** CoT exemplars verify against their own gold.
- **Zero Devanagari** remains anywhere in the tree.

## Korean conventions applied consistently

- **Script detection:** `"HANGUL" in unicodedata.name(c)` (Hangul syllables are `.isalpha()`).
- **Agglutinative overlap:** content-word matching strips trailing particles (조사) —
  `_strip_particle` + a Korean stop set — so 조선/조선은/조선이/조선을 collapse to one form.
  Used in `generate_grounded_ko`, `faithfulness_scorer`, `build_preference_data`, `self_consistency`.
- **Answer markers:** math `최종답:` (also 정답/답/####/\boxed/English/last-number);
  steps `단계 N:`; bootstrap delimiters `문제:` / `풀이:`; NLI `예` / `아니`; judge verdict `판정:`.
- **MCQ labels** normalized to A–E from circled `①②③④⑤`, Korean `가나다라마`, Latin `A–E`,
  and 1-based digits `1–5` (optional `번` suffix).
- **Field names** (exemplars / grounded TASKS): 단락 / 질문 / 답 / 풀이와 답 / 질문-답 /
  요약 / 제목 / 핵심사실 / 쉬운 설명 / 개요 / 용어 / 정의.
- **Resources** swapped to Korean: corpora 모두의 코퍼스 / AI Hub / KLUE; eval **KMMLU / KoBEST /
  HAERAE / CLIcK** (MILU → KMMLU). Eval language slice re-keyed `hi` → `ko`.

## Bugs fixed during the port

1. **`from_kmmlu.py`** — KMMLU answers are **1-based** (1–4 → A–D). The naive 0-based
   index used by `from_mmlu` would be off-by-one; the converter now maps 1-based correctly
   and round-trips through `verify_mcq`.
2. **`gen_format.py` list-count extraction** — Korean attaches a counter to the number
   (`3가지`, `4개`), so the old `tok.isdigit()` scan missed it. Switched to a `\d+` regex.
3. **`verify_mcq.py`** — full rewrite for Korean label systems (circled / 가나다라 / digit / Latin),
   `번` suffix, and `n_options` bounding.
4. **`test_tokenizer_merge.py`** — the original check ("token decodes to ≥2 codepoints")
   is Devanagari-specific (one akshara already spans 2+ codepoints). Hangul syllables are
   single precomposed codepoints, so the Korean-correct signal is "BPE composed a whole
   Hangul syllable from its bytes" — the check was updated accordingly.
5. **`eval_report.py`** — the target-language slice/default was hard-coded `"hi"`; re-keyed to `"ko"`.

## Notes carried over from the original review

- Stage-8 chain is consistent end-to-end: converters emit `n_options` → `translate_problems`
  preserves it → `rejection_sample` / `few_shot_cot_bootstrap` pass it to `verify_mcq`
  (signature-based kwarg dispatch).
- Stage-4c long-context retention is included (`build_long_docs.py`, `stage2c_longctx.py`,
  `needle_haystack.py`), and the CoT exemplar sets cover all five verifier formats.
