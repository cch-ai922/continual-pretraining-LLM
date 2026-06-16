#!/usr/bin/env python3
"""
Stage 5 (filter) — detect 높임말 / 문체 (speech-level / register) consistency
within a single Korean response.

Korean speech levels are morphologically marked at the sentence-final verb:
  합쇼체  — formal high   (-습니다, -습니까, -십시오)
  해요체  — informal polite (-요, -아요/어요, -이에요/예요, -세요)
  해체    — informal low  (-야/이야, -지, -네, -잖아, -거든, -더라, -구나)
  문어체  — literary       (sentence-final -다 in declarative prose:
                            한다/된다/있다/없다/이다/이었다/였다)

A response that mixes registers within itself is a defect: the playbook flags
this as D6 (rubric) and as a hard SFT-acceptance-gate item. Detection is
deterministic — Korean verb morphology is regular, so a suffix match on the
sentence-final eojeol is enough for the *consistency* question (the harder
*appropriateness* question — "is this register right for this prompt?" — is
NOT solved here; that needs a prompt-side label).

Three downstream uses:
  (a) Stage-5 SFT filter — drop training examples whose assistant turn mixes.
  (b) Stage-6 DPO signal — prefer the candidate with the higher consistency.
  (c) Stage-9 eval metric — report % register-consistent generations per ckpt.

Pure logic + JSONL driver + --selftest. No model, no labels needed.

Known limitations:
  * Bare 아/어/여 endings (e.g. "맛있어.") are NOT bucketed as 해체 because
    the same sound also appears in noun-final position (e.g. "단어"); without
    POS tags we'd false-positive on nouns. The cost: 해체 declaratives written
    in bare-verb form are bucketed as "neutral" instead. Chatbot data is almost
    entirely 해요체/합쇼체 anyway, so the practical hit is small.
  * Ambiguous bare 다 (literary 한다 vs casual 해체 -ㄴ다) is bucketed as
    문어체. This is wrong for spoken 해체 but inconsistency between 합쇼체
    and ANY -다 form is the failure case, and that still fires correctly.
"""
import argparse, json, re, sys


# ---------------------------------------------------------------------------
# SUFFIX TABLES — ordered most specific first within each bucket.
# Lookup priority across buckets: 합쇼체 > 해요체 > 문어체 > 해체.
# (해요체 ahead of 해체 because of 거든요/네요/지요 etc.; 문어체 ahead of
#  bare 해체 to keep -ㄴ다 in the 문어체 cluster.)
# ---------------------------------------------------------------------------
HAPSYO = (
    "습니다", "습니까",
    "십시오", "십시다",
    "읍시다",
    "ㄴ다고요",          # rare, but if present clearly formal
    "니다", "니까",     # catches 합니다 / 입니다 / 됩니다 / 갑니다 etc.
)

HAEYO = (
    "이에요", "예요", "이예요",
    "으세요", "으셔요", "세요", "셔요",
    "거든요", "는데요", "ㄴ데요",
    "지요", "네요", "군요", "데요",
    "아요", "어요", "여요", "와요", "워요",
    "에요",
    "요",                # bare polite ender — must be last in this bucket
)

MUNEO = (
    "이었다", "였다", "었다", "았다", "했다", "되었다",
    "한다", "된다", "있다", "없다", "이다",
    "는다", "ㄴ다",
    "다",                # bare — last; default 문어체 cluster
)

HAE = (
    "이야", "야",
    "잖아", "거든", "더라", "구나", "는구나",
    "는데", "ㄴ데",
    "지", "네",          # short — only fire AFTER 해요체 has been tried
)

# Short stand-alone interjection responses (so a one-word "네." is 해요체,
# a one-word "응." is 해체).
SHORT_FORM = {
    "네": "해요체",
    "예": "해요체",
    "그래요": "해요체",
    "아니요": "해요체",
    "아니에요": "해요체",
    "응": "해체",
    "그래": "해체",
    "아니": "해체",
    "아니야": "해체",
}

# Trailing characters to strip before suffix matching.
_TAIL_PUNCT = set("。.?!…」』\"')~")


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
def _strip_inline_noise(text):
    """Remove fenced code, inline code, embedded direct-speech quotes.

    Quoted speech often carries a different register from the wrapping prose
    ('"가자"라고 말했습니다' — quote is 해체, wrapper is 합쇼체). We want the
    wrapper's register, not the quote's, so the quote is scrubbed before
    sentence splitting.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"「[^」]*」", " ", text)
    text = re.sub(r"『[^』]*』", " ", text)
    text = re.sub(r'"[^"\n]{1,200}"', " ", text)
    text = re.sub(r"'[^'\n]{1,200}'", " ", text)
    return text


def split_sentences(text):
    """Split into sentence-like chunks on . ! ? 。 … or newline."""
    text = _strip_inline_noise(text)
    parts = re.split(r"[.!?。…\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_list_item(sent):
    """Short bullet/numbered/dashed items are register-neutral fragments."""
    s = sent.strip()
    if not s:
        return True
    if re.match(r"^[\-\*•]\s", s):                    # - foo  / • foo
        return len(s) < 60
    if re.match(r"^\d+[.)]\s", s):                          # 1. foo / 1) foo
        return len(s) < 60
    if re.match(r"^[①-⑳]\s*", s):                 # ①②③...
        return len(s) < 60
    return False


def classify_sentence(sent):
    """Classify one sentence into one of: 합쇼체 / 해요체 / 문어체 / 해체 / neutral."""
    s = sent.strip()
    while s and s[-1] in _TAIL_PUNCT:
        s = s[:-1]
    if not s:
        return "neutral"
    if _is_list_item(sent):
        return "neutral"
    if s in SHORT_FORM:
        return SHORT_FORM[s]

    for suf in HAPSYO:
        if s.endswith(suf):
            return "합쇼체"
    for suf in HAEYO:
        if s.endswith(suf):
            return "해요체"
    for suf in MUNEO:
        if s.endswith(suf):
            return "문어체"
    for suf in HAE:
        if s.endswith(suf):
            return "해체"
    return "neutral"


def register_distribution(text):
    """Return {bucket -> sentence count} for one response."""
    dist = {"합쇼체": 0, "해요체": 0, "문어체": 0, "해체": 0, "neutral": 0}
    for s in split_sentences(text):
        dist[classify_sentence(s)] += 1
    return dist


def consistency_score(text):
    """Return (score, dominant_bucket, n_register_sents).

    score = (max non-neutral bucket count) / (total non-neutral sentences).
    1.0 = perfectly single-register. n_register_sents = 0 → returns (1.0, None, 0).
    """
    dist = register_distribution(text)
    register_sents = sum(v for k, v in dist.items() if k != "neutral")
    if register_sents == 0:
        return 1.0, None, 0
    dominant = max((b for b in dist if b != "neutral"), key=lambda b: dist[b])
    return dist[dominant] / register_sents, dominant, register_sents


def is_consistent(text, threshold=0.8, min_sents=2):
    """A response is consistent if the dominant bucket covers >= threshold of
    its register-bearing sentences. Responses with fewer than `min_sents`
    register-bearing sentences are treated as consistent (not enough signal).
    """
    score, _, n = consistency_score(text)
    if n < min_sents:
        return True
    return score >= threshold


def matches_target_register(text, target):
    """True iff the response's dominant bucket equals `target`. Used when a
    generation request specified a register and we want to drop off-target
    output."""
    _, dominant, n = consistency_score(text)
    if n == 0:
        return False
    return dominant == target


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def _pick_text(row, mode):
    if mode == "text":
        return row.get("text", "")
    if mode == "assistant":
        for m in row.get("messages", []):
            if m.get("role") == "assistant":
                return m.get("content", "")
        return ""
    # auto
    if "text" in row:
        return row["text"]
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL with {text} OR chat-format {messages:[...]}")
    ap.add_argument("--out", default=None,
                    help="If given, write only kept rows to this file")
    ap.add_argument("--field", default="auto",
                    choices=["auto", "text", "assistant"],
                    help="which field to score (default: auto)")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--min-sents", type=int, default=2)
    ap.add_argument("--target", default=None,
                    choices=[None, "합쇼체", "해요체", "문어체", "해체"],
                    help="if set, also drop rows whose dominant bucket differs")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    stats = {"total": 0, "consistent": 0, "mixed": 0, "all_neutral": 0,
             "wrong_target": 0}
    dom_counts = {"합쇼체": 0, "해요체": 0, "문어체": 0, "해체": 0}

    for row in rows:
        stats["total"] += 1
        text = _pick_text(row, args.field)
        score, dominant, n = consistency_score(text)
        if n == 0:
            stats["all_neutral"] += 1
            keep = True
        else:
            dom_counts[dominant] += 1
            consistent = (n < args.min_sents) or (score >= args.threshold)
            if not consistent:
                stats["mixed"] += 1
                keep = False
            elif args.target and dominant != args.target:
                stats["wrong_target"] += 1
                keep = False
            else:
                stats["consistent"] += 1
                keep = True
        if out_f and keep:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if out_f:
        out_f.close()

    print(f"total            : {stats['total']:,}")
    print(f"consistent (kept): {stats['consistent']:,}")
    print(f"all-neutral (kept): {stats['all_neutral']:,}")
    print(f"mixed (dropped)  : {stats['mixed']:,}")
    if args.target:
        print(f"off-target (dropped): {stats['wrong_target']:,}")
    print("dominant-bucket counts (non-neutral rows):")
    for b, c in dom_counts.items():
        print(f"  {b}: {c:,}")
    if args.out:
        print(f"-> kept rows written to {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # pure 합쇼체
    t = "안녕하십니까. 저는 학생입니다. 만나서 반갑습니다."
    s, dom, n = consistency_score(t)
    assert dom == "합쇼체" and s == 1.0, (dom, s, n, register_distribution(t))

    # pure 해요체 (including 워요 bare-vowel + 세요 honorific + 이에요 copula)
    t = "안녕하세요. 저는 학생이에요. 만나서 반가워요."
    s, dom, n = consistency_score(t)
    assert dom == "해요체" and s == 1.0, (dom, s, n, register_distribution(t))

    # pure 해체 (이야 copula + 네 ender + 구나 ender)
    t = "나는 학생이야. 만나서 반갑네. 정말 즐겁구나."
    s, dom, n = consistency_score(t)
    assert dom == "해체" and s == 1.0, (dom, s, n, register_distribution(t))

    # pure 문어체 (literary -다)
    t = "이 글은 조선어를 다룬다. 조선어는 교착어이다. 어순은 SOV이다."
    s, dom, n = consistency_score(t)
    assert dom == "문어체" and s == 1.0, (dom, s, n, register_distribution(t))

    # MIXED — 합쇼체 + 해요체 (the canonical chatbot failure mode)
    t = "안녕하십니까. 저는 학생이에요. 만나서 반갑습니다."
    s, dom, n = consistency_score(t)
    assert 0 < s < 1 and n == 3, (s, n)
    assert not is_consistent(t, threshold=0.8)

    # MIXED — 합쇼체 + 해체
    t = "이것은 책입니다. 그리고 이건 펜이야."
    assert not is_consistent(t)

    # MIXED — 합쇼체 + 문어체 at 3:1, fails 0.8 threshold
    t = "이것은 책입니다. 저것은 펜입니다. 그것은 종이입니다. 그래 알겠다."
    assert not is_consistent(t, threshold=0.8)
    assert is_consistent(t, threshold=0.7)  # loosened threshold passes

    # Embedded direct-speech quote: the quote is 해체 ("내 책이야") but the
    # wrapper is 합쇼체. We want the wrapper.
    t = '그는 "그건 내 책이야"라고 말했습니다. 저는 동의했습니다.'
    s, dom, n = consistency_score(t)
    assert dom == "합쇼체" and n == 2, (dom, n, register_distribution(t))

    # Fenced code block must not perturb the wrapper's register
    t = "사용 방법입니다.\n```python\ndef f(): return 1\n```\n실행해 보십시오."
    s, dom, n = consistency_score(t)
    assert dom == "합쇼체", (dom, n, register_distribution(t))

    # Pure list — no register-bearing sentences, treated as consistent
    t = "- 사과\n- 배\n- 감"
    s, dom, n = consistency_score(t)
    assert n == 0 and dom is None
    assert is_consistent(t)

    # Single-sentence response is always "consistent enough" with min_sents=2
    assert is_consistent("안녕하십니까.")
    assert is_consistent("그래.")

    # Short interjection forms route to the right bucket
    assert classify_sentence("네") == "해요체"
    assert classify_sentence("응") == "해체"
    assert classify_sentence("아니에요") == "해요체"
    assert classify_sentence("아니야") == "해체"

    # Target-register matching
    assert matches_target_register("안녕하세요. 학생이에요.", "해요체")
    assert not matches_target_register("안녕하세요. 학생이에요.", "합쇼체")

    # Chat-format extraction
    chat = {"messages": [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": "안녕하세요. 반가워요."}]}
    assert _pick_text(chat, "auto") == "안녕하세요. 반가워요."
    assert _pick_text(chat, "assistant") == "안녕하세요. 반가워요."

    # 거든요 / 네요 / 군요 must be 해요체, not 해체 (priority order matters)
    assert classify_sentence("그래서 갔거든요") == "해요체"
    assert classify_sentence("정말 좋네요") == "해요체"
    assert classify_sentence("멋지군요") == "해요체"

    # 한다 / 된다 / 이다 stay in 문어체 even though they share -다 with 해체
    assert classify_sentence("그는 매일 공부한다") == "문어체"
    assert classify_sentence("이것이 정의이다") == "문어체"

    # 안녕하십시오 → imperative 합쇼체
    assert classify_sentence("안녕히 가십시오") == "합쇼체"

    print("PASS register_consistency: 4 pure registers + mixed-register cases + "
          "quote/code/list scrubbing + short-form + priority order")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
