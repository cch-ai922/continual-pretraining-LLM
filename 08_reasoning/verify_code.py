#!/usr/bin/env python3
"""
Stage 8 (verifier) — language-INDEPENDENT correctness check for Python code.

For RLVR on code (HumanEval / MBPP / APPS / LiveCodeBench-style):
  * the problem statement can be in Korean (translated)
  * the model's generated code is Python (universal)
  * the unit tests are Python (universal)
  * passing all tests = correct, regardless of how the model reasoned

Extraction: looks for ```python ... ``` blocks first, then any ``` ... ``` block,
then falls back to treating the whole response as code.

⚠️  SANDBOX WARNING: this runs UNTRUSTED MODEL-GENERATED CODE in a Python subprocess
with a timeout but NO resource isolation. For production-scale RLVR, wrap the
subprocess in a real sandbox (firejail / gVisor / nsjail / docker), block network,
and limit memory/CPU. The timeout here protects against infinite loops but NOT
against fork-bombs, file-system writes, or malicious imports.
"""
import argparse, json, os, re, subprocess, sys, tempfile


# ---------------------------------------------------------------------------
# PURE EXTRACTION (testable without execution)
# ---------------------------------------------------------------------------
_FENCED = re.compile(r"```(\w*)\s*\n(.*?)```", re.S)


def extract_code(text: str, language: str = "python") -> str:
    """Pull the code out of the model's response.

    Priority (each tried last-to-first within the response, since models tend to
    explain first and emit the solution last):
        1. block tagged with the requested language (e.g. ```python)
        2. untagged block (```)
        3. any block, regardless of tag
        4. the whole response as-is, if there are no fences at all
    """
    if not text:
        return ""
    blocks = list(_FENCED.finditer(text))
    if not blocks:
        return text.strip()
    aliases = {language.lower()} | ({"py"} if language == "python" else set())
    for m in reversed(blocks):
        if m.group(1).lower() in aliases:
            return m.group(2).strip()
    for m in reversed(blocks):
        if m.group(1) == "":
            return m.group(2).strip()
    return blocks[-1].group(2).strip()


# ---------------------------------------------------------------------------
# EXECUTION (subprocess + timeout)
# ---------------------------------------------------------------------------
def run_tests(code: str, tests: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Concatenate code + tests, run in a subprocess. Return (passed, message).

    'passed' is True iff the subprocess exits with code 0 (i.e. no assertion fired
    and no exception was raised). The message contains stderr/stdout for debug.
    """
    if not code:
        return False, "EMPTY_CODE"
    full = code.rstrip() + "\n\n" + tests
    fd, path = tempfile.mkstemp(suffix=".py", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(full)
        r = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        msg = (r.stderr or r.stdout or "").strip()
        return r.returncode == 0, msg[:400]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT (>{timeout}s)"
    except Exception as e:
        return False, f"EXEC_ERROR: {e}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# UNIFORM VERIFIER INTERFACE  (matches verify_math.verify signature)
# ---------------------------------------------------------------------------
def verify(model_output: str, gold) -> bool:
    """Plug into rejection_sample.py / grpo_train.py.

    `gold` is either:
      - a string of Python test code (e.g. 'assert add(2,3) == 5'), OR
      - a dict {"tests": "...", "timeout": 5.0, "setup": "import math"}.
    """
    if isinstance(gold, dict):
        tests = gold.get("tests", "")
        timeout = float(gold.get("timeout", 5.0))
        setup = gold.get("setup", "")
        code = extract_code(model_output)
        if setup:
            code = setup.rstrip() + "\n" + code
    else:
        tests = str(gold or "")
        timeout = 5.0
        code = extract_code(model_output)
    passed, _ = run_tests(code, tests, timeout)
    return passed


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True,
                    help="JSONL with {prediction/response, tests/gold}")
    ap.add_argument("--out", default=None, help="optional: per-row verdicts JSONL")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    n = ok = 0
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    for line in open(args.inp, encoding="utf-8"):
        r = json.loads(line); n += 1
        pred = r.get("prediction") or r.get("response") or r.get("output", "")
        gold = r.get("tests") or r.get("gold") or r.get("answer") or ""
        if isinstance(gold, str) and not isinstance(r.get("gold"), dict):
            v = verify(pred, gold)
        else:
            v = verify(pred, r.get("gold", gold))
        ok += int(v)
        if fout:
            fout.write(json.dumps({**r, "verified": v}, ensure_ascii=False) + "\n")
    if fout:
        fout.close()
    print(f"verified {ok:,}/{n:,}  ({ok/max(n,1):.1%})")


# ---------------------------------------------------------------------------
def _selftest():
    # extraction
    assert extract_code("```python\ndef f(): return 1\n```") == "def f(): return 1"
    assert extract_code("text\n```\nx = 1\n```\nmore") == "x = 1"
    assert extract_code("no fences, just\ndef f(): pass") == "no fences, just\ndef f(): pass"
    # multiple blocks: take LAST (model explains then emits solution)
    assert extract_code("```python\nold = 1\n```\nlater:\n```python\nfinal = 2\n```") == "final = 2"
    # language-mismatched fence falls back to last "any" fence
    assert extract_code("```bash\nls\n```\n```\nz=3\n```") == "z=3"
    assert extract_code("") == ""

    # execution: passing test
    ok, _ = run_tests("def add(a, b): return a + b", "assert add(2, 3) == 5")
    assert ok, "good code+test should pass"
    # failing test
    fail, msg = run_tests("def add(a, b): return a + b", "assert add(2, 3) == 6")
    assert not fail and "AssertionError" in msg
    # syntax error
    fail, msg = run_tests("def add( :::", "")
    assert not fail and ("SyntaxError" in msg or "invalid" in msg.lower())
    # timeout
    fail, msg = run_tests("while True: pass", "", timeout=0.4)
    assert not fail and "TIMEOUT" in msg
    # empty code
    assert run_tests("", "assert True") == (False, "EMPTY_CODE")

    # end-to-end verify
    sample_good = "```python\ndef double(x): return x * 2\n```"
    assert verify(sample_good, "assert double(7) == 14")
    assert not verify(sample_good, "assert double(7) == 15")
    # dict gold form
    assert verify(sample_good, {"tests": "assert double(3) == 6", "timeout": 3.0})

    print("PASS all code-verifier tests")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
