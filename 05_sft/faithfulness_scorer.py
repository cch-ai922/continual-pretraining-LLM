#!/usr/bin/env python3
"""
Stage 5 (filter) — score faithfulness of grounded generations using the STAGE-4
BASE model as a SCORER (no instruction-following required).

Two complementary model-based signals, plus the cheap lexical pre-filter:

  PMI grounding score :  logP(answer | passage)  -  logP(answer | NO passage)
      A faithful answer is much more likely WITH the passage than without it.
      Robust to paraphrase (unlike lexical overlap). Pure forward passes.

  Few-shot NLI        :  show labeled (passage, claim, 예/아니오) exemplars, then
      compare the base model's logprob of " 예" vs " 아니오" for the new pair.
      Leans on entailment ability learned in English, transferred to Korean.

Gate = lexical pre-filter AND (PMI clears tau_pmi OR NLI says supported). Calibrate
the thresholds on ~150 hand-labeled generations; PREFER PRECISION (dropping good
data is fine; keeping hallucinations teaches the model to hallucinate).

Pure decision/prompt logic is unit-tested: run with --selftest.
"""
import argparse, json, re, unicodedata

DELIM = "\n---\n"

DEFAULT_NLI_EXEMPLARS = [
    {"passage": "대동강은 조선의 주요강으로서 랑림산맥에서 발원한다.",
     "claim": "대동강은 랑림산맥에서 발원한다.", "label": "예"},
    {"passage": "안학궁은 평양에 있으며 고구려시기에 건설되였다.",
     "claim": "안학궁은 함흥에 있다.", "label": "아니"},
    {"passage": "빛합성에서 식물은 산소를 방출한다.",
     "claim": "빛합성에서 식물은 산소를 내보낸다.", "label": "예"},
]


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
_PARTICLES = sorted([
    "입니다", "이였다", "이며", "이고", "이다",   # fused copula endings (noun+이다)
    "에서는", "으로는", "에게서", "께서는", "이라고", "이라는",
    "에서", "에게", "께서", "으로", "라고", "부터", "까지", "보다", "처럼",
    "만큼", "마다", "조차", "마저", "밖에", "이나", "든지", "이라", "에는",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "만",
    "로", "야", "께", "라", "나",
], key=len, reverse=True)


def _strip_particle(w):
    for p in _PARTICLES:
        if len(w) - len(p) >= 2 and w.endswith(p):   # keep >=2-syll stem (정의/수도 intact)
            return w[: -len(p)]
    return w


_KO_STOP = {"그리고", "그러나", "하지만", "또한", "그래서", "따라서", "즉", "및",
            "등", "것", "수", "더", "매우", "아주", "이", "그", "저", "때문",
            "하다", "있다", "없다", "되다", "이다", "한다", "합니다", "입니다",
            "있습니다", "같은", "위하여", "통해", "대한", "대하여"}


def korean_fraction(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "HANGUL" in unicodedata.name(c, "")) / len(letters)


def _content_words(text):
    out = set()
    for w in text.split():
        w = w.strip("。.,?!\"'()[]{}:;…·")
        w = _strip_particle(w)
        if len(w) >= 2 and korean_fraction(w) > 0.5 and w not in _KO_STOP:
            out.add(w)
    return out


def lexical_overlap(answer, passage):
    src = _content_words(passage)
    return 1.0 if not src else len(src & _content_words(answer)) / len(src)


def pmi_score(logp_with_passage, logp_without_passage):
    """Grounding score: how much the passage raises the answer's log-likelihood."""
    return logp_with_passage - logp_without_passage


def pmi_decision(score, tau=2.0):
    return score >= tau


def nli_prompt(passage, answer, exemplars=DEFAULT_NLI_EXEMPLARS):
    blocks = [f"단락: {e['passage']}\n주장: {e['claim']}\n"
              f"주장이 단락에서 뒤받침됩니까? 답: {e['label']}" for e in exemplars]
    blocks.append(f"단락: {passage}\n주장: {answer}\n"
                  f"주장이 단락에서 뒤받침됩니까? 답:")
    return DELIM.join(blocks)


def nli_decision(logp_yes, logp_no, margin=0.5):
    """Supported iff the model prefers ' 예' over ' 아니' by at least `margin` nats."""
    return (logp_yes - logp_no) >= margin


def is_faithful(lex, pmi, nli_yes, nli_no, tau_lex=0.05, tau_pmi=2.0, margin=0.5):
    if lex < tau_lex:
        return False                       # cheap pre-filter: no topical overlap at all
    return pmi_decision(pmi, tau_pmi) or nli_decision(nli_yes, nli_no, margin)


# ---------------------------------------------------------------------------
# BASE-MODEL SCORER  (the only model-dependent part)
# ---------------------------------------------------------------------------
class BaseScorer:
    """Wraps the Stage-4 base model to return summed token log-probs of a
    continuation given a prompt. HF implementation; a vLLM `prompt_logprobs`
    path is faster for batch jobs (see note)."""

    def __init__(self, model_path):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True).eval()

    def logprob(self, prompt, continuation):
        torch = self.torch
        ids_p = self.tok(prompt, return_tensors="pt").input_ids
        ids_f = self.tok(prompt + continuation, return_tensors="pt").input_ids.to(self.model.device)
        n_cont = ids_f.shape[1] - ids_p.shape[1]
        if n_cont <= 0:
            return 0.0
        with torch.no_grad():
            logits = self.model(ids_f).logits[0, :-1]      # predict t+1 from t
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids_f[0, 1:]
        tok_lp = lp[torch.arange(tgt.shape[0]), tgt]
        return tok_lp[-n_cont:].sum().item()

    def score_pair(self, passage, answer):
        with_p = self.logprob(f"단락: {passage}\n답: ", answer)
        without_p = self.logprob("답: ", answer)
        prompt = nli_prompt(passage, answer)
        yes = self.logprob(prompt, " 예")
        no = self.logprob(prompt, " 아니")
        return pmi_score(with_p, without_p), yes, no


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Stage-4 base model (HF dir)")
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL with {passage, answer} OR chat {messages:[user(passage),assistant]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-pmi", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--min-korean", type=float, default=0.7)
    args = ap.parse_args()

    scorer = BaseScorer(args.model)
    kept = dropped = 0
    fout = open(args.out, "w", encoding="utf-8")
    fflag = open(args.out + ".rejected.jsonl", "w", encoding="utf-8")

    for line in open(args.inp, encoding="utf-8"):
        ex = json.loads(line)
        if "messages" in ex:
            passage = ex["messages"][0]["content"]
            answer = ex["messages"][-1]["content"]
        else:
            passage, answer = ex["passage"], ex["answer"]

        lex = lexical_overlap(answer, passage)
        pmi, yes, no = scorer.score_pair(passage, answer)
        ok = (korean_fraction(answer) >= args.min_korean and
              is_faithful(lex, pmi, yes, no, tau_pmi=args.tau_pmi, margin=args.margin))
        rec = dict(ex, _scores={"lex": round(lex, 3), "pmi": round(pmi, 2),
                                "nli_margin": round(yes - no, 2)})
        (fout if ok else fflag).write(json.dumps(rec, ensure_ascii=False) + "\n")
        kept += ok; dropped += (not ok)

    fout.close(); fflag.close()
    print(f"kept {kept:,} | rejected {dropped:,} -> {args.out} (+ .rejected.jsonl)")


# ---------------------------------------------------------------------------
def _selftest():
    # NLI margin
    assert nli_decision(-0.1, -3.0, 0.5) is True
    assert nli_decision(-3.0, -0.1, 0.5) is False
    # PMI
    assert abs(pmi_score(-5.0, -12.0) - 7.0) < 1e-9
    assert pmi_decision(7.0, 2.0) and not pmi_decision(0.5, 2.0)
    # combined gate
    assert is_faithful(lex=0.4, pmi=7.0, nli_yes=-0.1, nli_no=-3.0)   # both fire
    assert is_faithful(lex=0.4, pmi=0.0, nli_yes=-0.1, nli_no=-3.0)   # NLI alone
    assert is_faithful(lex=0.4, pmi=7.0, nli_yes=-3.0, nli_no=-0.1)   # PMI alone
    assert not is_faithful(lex=0.0, pmi=99, nli_yes=0, nli_no=-99)    # no overlap -> reject
    # prompt shape
    p = nli_prompt("어떤 단락", "어떤 주장")
    assert p.rstrip().endswith("답:") and "어떤 단락" in p
    # lexical overlap with particle normalization (안학궁은 ~ 안학궁이)
    assert lexical_overlap("안학궁은 평양에 있다.", "안학궁이 평양에 있다.") > 0.3
    print("PASS all faithfulness-scorer tests")


if __name__ == "__main__":
    import sys
    _selftest() if "--selftest" in sys.argv else main()
