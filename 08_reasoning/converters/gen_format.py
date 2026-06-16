#!/usr/bin/env python3
"""Generate synthetic Korean format-following prompts → {problem, gold} for verify_format.

IFEval-style: produce prompts that REQUEST a specific structured output, with the
gold being the format spec. The verifier (verify_format.verify) checks structural
adherence, not content correctness — the right reward signal for teaching format.

Four format types covered (matching verify_format.py):
  json      — request a JSON object with required keys / typed fields
  list      — request N bullets/numbered items
  md_table  — request a markdown table with specified columns
  regex     — request a response matching a specified pattern
"""
import argparse, json, random, re, sys


ENTITIES = {
    "people":  ["세종왕", "리순신", "김구", "류관순"],
    "places":  ["평양", "개성", "개천", "함흥", "평성"],
    "topics":  ["기후변화", "건강한 생활습관", "정보기술교육", "생물다양성"],
    "animals": ["호랑이", "까치", "풍산개", "고래", "사슴"],
    "foods":   ["비빔밥", "김치", "불고기", "떡국", "김밥"],
}

# ----- JSON templates ------------------------------------------------------
JSON_TEMPLATES = [
    ("{person}에 대한 JSON객체를 만드시오. 다음 열쇠를 포함해야 합니다: "
     "'이름'(string), '출생년도'(integer), '업적'(string).",
     {"required": ["이름", "출생년도", "업적"],
      "types": {"이름": "str", "출생년도": "int", "업적": "str"}}),
    ("{place}에 대한 JSON객체를 주시오. '도시'(string), "
     "'인구'(integer), '특징'(list)을 포함시키시오.",
     {"required": ["도시", "인구", "특징"],
      "types": {"도시": "str", "인구": "int", "특징": "list"}}),
]
# ----- LIST templates ------------------------------------------------------
LIST_TEMPLATES = [
    ("{topic}에 대한 주요사항을 {n}가지 목록으로 작성하시오.", None),
    ("{place}{place_obj} 방문해야 하는 리유를 {n}가지 항목으로 쓰시오.", None),
    ("{animal}에 대한 흥미로운 사실을 {n}개 목록으로 주시오.", None),
]
# ----- MD TABLE templates --------------------------------------------------
TABLE_TEMPLATES = [
    ("유명한 조선인물 세명에 대한 markdown표를 만드시오. 렬: "
     "이름, 출생년도, 업적.",
     {"columns": ["이름", "출생년도", "업적"], "min_rows": 3}),
    ("조선의 네개 도시와 그 별칭에 대한 markdown표를 주시오. "
     "렬: 도시, 별칭.",
     {"columns": ["도시", "별칭"], "min_rows": 4}),
]
# ----- REGEX templates -----------------------------------------------------
REGEX_TEMPLATES = [
    ("{topic}에 대해 아래의 구조를 정확히 따라 한문장으로 쓰시오: "
     "'주제: <X> | 중요성: <Y>'",
     r"주제\s*:\s*\S.*?\|\s*중요성\s*:\s*\S"),
    ("{food}에 대해 다음 형식으로 답하시오: '료리: <이름>. "
     "주원료: <목록>. 유래: <지역>.'",
     r"료리\s*:\s*\S.*?주원료\s*:\s*\S.*?유래\s*:\s*\S"),
]


def _has_batchim(ch: str) -> bool:
    """True if a Hangul syllable ends in a final consonant (받침)."""
    if not ("\uac00" <= ch <= "\ud7a3"):
        return False  # non-Hangul (e.g. a digit/letter): treat as vowel-final
    return (ord(ch) - 0xAC00) % 28 != 0


def _josa(word: str, after_consonant: str, after_vowel: str) -> str:
    """Pick the grammatically correct Korean particle for `word`.

    Korean object/subject/topic particles vary with the final sound of the noun:
    을/를, 이/가, 은/는, 과/와. Hard-coding one form produces 평양'를' (wrong;
    평양 ends in ㄹ → 평양'을'), so choose by batchim.
    """
    return after_consonant if (word and _has_batchim(word[-1])) else after_vowel


def _fill(template: str, rng: random.Random) -> str:
    person = rng.choice(ENTITIES["people"])
    place  = rng.choice(ENTITIES["places"])
    topic  = rng.choice(ENTITIES["topics"])
    animal = rng.choice(ENTITIES["animals"])
    food   = rng.choice(ENTITIES["foods"])
    return template.format(
        person=person, place=place, topic=topic, animal=animal, food=food,
        n=rng.choice([3, 4, 5]),
        # batchim-correct particles for each entity (used as {place_obj} etc.)
        place_obj=_josa(place, "을", "를"), place_sub=_josa(place, "은", "는"),
        place_nom=_josa(place, "이", "가"),
        person_obj=_josa(person, "을", "를"), animal_obj=_josa(animal, "을", "를"),
        food_obj=_josa(food, "을", "를"), topic_obj=_josa(topic, "을", "를"),
    )


def generate_one(ftype: str, rng: random.Random) -> dict:
    if ftype == "json":
        tpl, schema = rng.choice(JSON_TEMPLATES)
        return {"problem": _fill(tpl, rng), "gold": {"type": "json", "schema": schema}}
    if ftype == "list":
        tpl, _ = rng.choice(LIST_TEMPLATES)
        prompt = _fill(tpl, rng)
        # Korean attaches a counter to the number ("3가지", "4개"), so tok.isdigit()
        # would miss it — pull the first run of digits instead.
        m = re.search(r"\d+", prompt)
        n = int(m.group()) if m else 3
        return {"problem": prompt, "gold": {"type": "list", "min_items": n}}
    if ftype == "md_table":
        tpl, spec = rng.choice(TABLE_TEMPLATES)
        return {"problem": _fill(tpl, rng), "gold": {"type": "md_table", **spec}}
    if ftype == "regex":
        tpl, pat = rng.choice(REGEX_TEMPLATES)
        return {"problem": _fill(tpl, rng),
                "gold": {"type": "regex", "pattern": pat, "ignore_case": False}}
    raise ValueError(f"unknown format type: {ftype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--types", nargs="+", default=["json", "list", "md_table", "regex"],
                    choices=["json", "list", "md_table", "regex"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            t = rng.choice(args.types)
            f.write(json.dumps(generate_one(t, rng), ensure_ascii=False) + "\n")
    print(f"wrote {args.n:,} prompts ({'+'.join(args.types)}) -> {args.out}")


def _selftest():
    rng = random.Random(0)
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from verify_format import verify

    # JSON
    j = generate_one("json", rng)
    keys = j["gold"]["schema"]["required"]; types = j["gold"]["schema"]["types"]
    obj = {k: ("x" if types.get(k) == "str" else 1 if types.get(k) == "int" else []) for k in keys}
    good = "여기 있습니다: " + json.dumps(obj, ensure_ascii=False)
    assert verify(good, j["gold"])
    assert not verify("not json at all", j["gold"])

    # LIST — the {n}가지 / {n}개 counter must be parsed correctly
    li = generate_one("list", rng)
    n = li["gold"]["min_items"]
    assert n in (3, 4, 5), n                         # proves the digit-run fix works
    response = "\n".join(f"- 항목 {i+1}" for i in range(n + 1))
    assert verify(response, li["gold"])
    assert not verify("- 하나뿐", li["gold"])

    # MD TABLE
    tbl = generate_one("md_table", rng)
    cols = tbl["gold"]["columns"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join("x" for _ in cols) + " |" for _ in range(tbl["gold"]["min_rows"])]
    assert verify("\n".join([header, sep] + rows), tbl["gold"])

    # REGEX
    rg = generate_one("regex", rng)
    sample = ("주제: 기후변화 | 중요성: 매우 큼" if "주제" in rg["problem"]
              else "료리: 김밥. 주원료: 밥, 김. 유래: 조선.")
    assert verify(sample, rg["gold"])
    assert not verify("전혀 다른 문장.", rg["gold"])
    print("PASS format generator (incl. round-trip through verify_format for all 4 types)")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
