#!/usr/bin/env python3
"""
Stage 7 (teacher loop) — SELF-CONSISTENCY for non-verifiable CoT data.

Wang et al. 2022 — for any prompt, sample K responses at high temperature; if
multiple samples *converge* to the same answer, that's evidence the model is
reasoning correctly rather than guessing. The convergent responses become SFT data.

For verifiable tasks we use `rejection_sample.py` with an external verifier.
For non-verifiable tasks (advice, summary, explanation, opinion) there is no
verifier — but consistency across samples is a usable proxy:

  * High consistency  → the model has a stable view of the answer       (KEEP)
  * Low consistency   → the model is guessing / hallucinating            (DROP)
  * Medium           → ambiguous prompt or genuine multi-valid-answer    (DROP)

How "convergence" is measured: cluster the responses by extracting their FINAL
ANSWER (the conclusion sentence(s)), then computing Jaccard similarity of content
characters. The largest cluster above `--min-cluster-frac` keeps its members
as SFT data; the original (full CoT, not just the answer) is preserved.

The strength of this signal scales with K. K=8 catches most hallucinations; K=4
is the practical minimum; K<3 is unreliable.
"""
import argparse, json, re, sys, unicodedata


# ---------------------------------------------------------------------------
# PURE, TESTABLE CORE
# ---------------------------------------------------------------------------
_FINAL_MARKERS = ["최종답", "결론", "따라서", "그러므로", "요약",
                  "final answer", "therefore", "in conclusion"]


def extract_final(response: str, fallback_chars: int = 300) -> str:
    """Pull out the answer / conclusion portion of a CoT response. We look for an
    explicit marker; if none, take the last `fallback_chars` characters (the model
    usually puts the conclusion at the end)."""
    if not response:
        return ""
    last_idx = -1
    for m in _FINAL_MARKERS:
        idx = response.rfind(m)
        if idx > last_idx:
            last_idx = idx
    if last_idx >= 0:
        return response[last_idx:].strip()
    return response[-fallback_chars:].strip()


_PARTICLES = sorted([
    "입니다", "이였다", "이며", "이고", "이다",   # fused copula endings (noun+이다)
    "에서는", "으로는", "에게서", "께서는", "이라고", "이라는",
    "에서", "에게", "께서", "으로", "라고", "부터", "까지", "보다", "처럼",
    "만큼", "마다", "조차", "마저", "밖에", "이나", "든지", "이라", "에는",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "만",
    "로", "야", "께", "라", "나",
], key=len, reverse=True)


def _strip_particle(w):
    for part in _PARTICLES:
        if len(w) - len(part) >= 2 and w.endswith(part):   # keep >=2-syll stem
            return w[: -len(part)]
    return w


_KO_STOP = {"그리고", "그러나", "하지만", "또한", "그래서", "따라서", "즉", "및",
            "등", "것", "수", "더", "매우", "아주", "이", "그", "저", "때문",
            "하다", "있다", "없다", "되다", "이다", "한다", "합니다", "입니다",
            "있습니다", "같은", "위해", "통해", "대한", "대해"}


def _content_tokens(text: str) -> set:
    """Content-token set (Korean-aware). Mirrors faithfulness_scorer / build_preference_data:
    strip the trailing particle and drop function words so josa-only overlap doesn't
    falsely group dissimilar conclusions. Numbers are kept (key for math conclusions)."""
    out = set()
    for w in text.split():
        w = w.strip("。.,?!\"'()[]{}:;…·")
        w = _strip_particle(w)
        if len(w) >= 2 and w not in _KO_STOP:
            out.add(w)
    return out


def similarity(a: str, b: str) -> float:
    A, B = _content_tokens(a), _content_tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def cluster_responses(finals, sim_thresh: float = 0.4):
    """Greedy single-link clustering on similarity ≥ sim_thresh.

    Returns: list of clusters, each a list of indices into `finals`. Greedy is
    order-dependent (a more principled approach is agglomerative or DBSCAN), but
    for K≤16 the difference is rarely meaningful and greedy is dead simple.
    """
    clusters = []
    for i, f in enumerate(finals):
        placed = False
        for cluster in clusters:
            rep_idx = cluster[0]
            if similarity(f, finals[rep_idx]) >= sim_thresh:
                cluster.append(i); placed = True; break
        if not placed:
            clusters.append([i])
    return clusters


def best_cluster(clusters, k_total: int, min_frac: float = 0.5):
    """Return the indices of the largest cluster if its size / K ≥ min_frac, else None."""
    if not clusters:
        return None
    best = max(clusters, key=len)
    return best if len(best) / k_total >= min_frac else None


# ---------------------------------------------------------------------------
# MODEL HOOK
# ---------------------------------------------------------------------------
def chat(messages_batch, model_path, k=8, max_new_tokens=512, temperature=0.9):
    """Sample K completions per prompt. Same shape as rejection_sample.chat —
    returns list[list[str]], k strings per prompt. See that file for the vLLM template."""
    raise NotImplementedError("Connect your model in chat(); return list[list[str]] (k per prompt).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL: {prompt}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="samples per prompt")
    ap.add_argument("--sim-thresh", type=float, default=0.4,
                    help="Jaccard threshold for grouping responses into the same cluster")
    ap.add_argument("--min-cluster-frac", type=float, default=0.5,
                    help="largest cluster must contain ≥ this fraction of K samples")
    ap.add_argument("--keep-per-prompt", type=int, default=1,
                    help="how many responses from the majority cluster to emit as SFT data")
    ap.add_argument("--system", default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts, encoding="utf-8")]
    msgs = [([{"role": "system", "content": args.system}] if args.system else []) +
            [{"role": "user", "content": p}] for p in prompts]

    kept = 0
    dropped_low = dropped_split = 0
    fout = open(args.out, "w", encoding="utf-8")
    for i in range(0, len(prompts), args.batch):
        batch_msgs = msgs[i:i + args.batch]
        batch_prompts = prompts[i:i + args.batch]
        samples_per = chat(batch_msgs, args.model, k=args.k)
        for p, samples in zip(batch_prompts, samples_per):
            finals = [extract_final(s) for s in samples]
            clusters = cluster_responses(finals, args.sim_thresh)
            best = best_cluster(clusters, args.k, args.min_cluster_frac)
            if best is None:
                # diagnose whether it's no-cluster (all different) or split (no majority)
                if max(len(c) for c in clusters) <= 1:
                    dropped_low += 1
                else:
                    dropped_split += 1
                continue
            # keep the shortest few from the majority cluster (shorter = clearer)
            best_sorted = sorted(best, key=lambda idx: len(samples[idx]))
            for idx in best_sorted[:args.keep_per_prompt]:
                fout.write(json.dumps({"messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": samples[idx].strip()}]},
                                      ensure_ascii=False) + "\n")
                kept += 1
    fout.close()
    print(f"kept {kept:,}  | dropped (no consensus: low={dropped_low}, split={dropped_split})  "
          f"-> {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    # extract_final picks the conclusion section when a marker exists
    r = ("먼저 문제를 리해한다. 거리 = 속도 × 시간. "
         "따라서 거리 = 60 × 2.5 = 150키로메터.")
    f = extract_final(r)
    assert "150키로메터" in f and "따라서" in f
    # similarity catches near-paraphrases of the same conclusion
    a = "따라서 거리는 150키로메터."
    b = "그러므로 거리는 150키로메터."
    c = "그러므로 거리는 200키로메터."            # different number
    assert similarity(a, b) > 0.3
    assert similarity(a, c) < similarity(a, b)
    # clustering: 4 samples, 3 agree on 150 and 1 disagrees -> majority cluster size 3
    finals = ["따라서 거리는 150키로메터.",
              "그러므로 거리는 150키로메터.",
              "최종답: 150키로메터",
              "최종답: 200키로메터"]
    clusters = cluster_responses(finals, sim_thresh=0.3)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3], sizes
    best = best_cluster(clusters, k_total=4, min_frac=0.5)
    assert best is not None and len(best) == 3
    # all-different responses -> no majority cluster
    all_diff = ["따라서 수도는 평양이다.",
                "그러므로 답은 함흥이다.",
                "최종답은 평성.",
                "따라서 정답은 청진."]
    clusters2 = cluster_responses(all_diff, sim_thresh=0.3)
    assert best_cluster(clusters2, 4, 0.5) is None
    # extract_final fallback when no marker is present
    no_marker = "여러가지 이야기, 더 많은 이야기, 마지막에 무엇인가 말했다."
    assert extract_final(no_marker, fallback_chars=20).endswith("말했다.")
    print("PASS self-consistency tests")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
