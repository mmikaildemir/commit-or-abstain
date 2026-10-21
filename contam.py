"""Build contamination-test variants of CaseHOLD test items (original condition).

For a fixed random sample of the full test split, one LLM call per item returns
  cited: the exact citation of the case whose holding is masked (text right before "(<HOLDING>)")
  anonymized: the whole passage with every case name and reporter citation replaced by plausible
              fictitious ones (Arvin 2025's anonymization), everything else verbatim
Code validates both, then derives three variants per item:
  anon      - anonymized passage                                   (Arvin's test)
  cite_only - just the cited case's citation + "(<HOLDING>)"       (does the model know the case?)
  no_cite   - passage with that citation replaced by "[citation omitted]" (context only)

Usage: CASEHOLD_SET=full python contam.py build [n=500] [seed=11]
Writes data/contam.jsonl (one row per valid item, same fields as full.jsonl plus "variants").
Run models on a variant with: CASEHOLD_SET=contam_<variant> python run.py MODEL original
(build also writes data/contam_<variant>.jsonl with the variant text in "prompt").
"""
import concurrent.futures as cf
import difflib
import json
import random
import re
import sys

import run

REWRITER = "openai/gpt-6-luna"
TASK = """You are preparing a legal NLP robustness test. Below is a passage from a US court opinion. The parenthetical holding of one cited case has been replaced by the token <HOLDING>.

Return JSON with two fields:
1. "cited": the exact citation of the case whose holding is masked, copied character-for-character from the passage: the text that immediately precedes "(<HOLDING>)", starting at the case name (or short-form name) and ending with its last parenthetical before <HOLDING> (court/year) if present. Do not include signals such as "See", "see also", "cf." or a leading semicolon.
2. "anonymized": the full passage with EVERY case name and EVERY reporter citation (volume, reporter, page, pincite, court and year parentheticals) replaced by plausible but fictitious ones. Keep the same citation format and style. Keep statutes, rules, legal reasoning, and all other words exactly as they are. Keep the token <HOLDING> exactly once, in the same place.

Passage:
{passage}"""

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["cited", "anonymized"],
          "properties": {"cited": {"type": "string"}, "anonymized": {"type": "string"}}}


def rewrite(item):
    body = {"model": REWRITER, "temperature": 0, "reasoning": {"effort": "low"},
            "messages": [{"role": "user", "content": TASK.format(passage=item["prompt"])}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "anon", "strict": True, "schema": SCHEMA}},
            "usage": {"include": True}}
    d = run.post("v1/chat/completions", body)
    out = json.loads(d["choices"][0]["message"]["content"])
    return out, d.get("usage", {}).get("cost") or 0


def validate(item, out):
    p, cited, anon = item["prompt"], out["cited"].strip(), out["anonymized"]
    h = p.find("(<HOLDING>)")
    if not cited or h < 0:
        return None, "no_holding_or_cite"
    # the citation must sit right before "(<HOLDING>)" (allowing whitespace between)
    before = p[:h].rstrip()
    if not before.endswith(cited.rstrip()):
        return None, "cite_not_adjacent"
    if anon.count("<HOLDING>") != 1:
        return None, "anon_holding_count"
    sim = difflib.SequenceMatcher(None, p, anon).ratio()
    if sim < 0.6:
        return None, "anon_too_different"
    start = len(before) - len(cited.rstrip())
    return {"anon": anon,
            "cite_only": f"{cited} (<HOLDING>)",
            "no_cite": p[:start] + "[citation omitted]" + p[len(before):],
            "cited": cited, "anon_similarity": round(sim, 3)}, None


def build(n, seed):
    items = [json.loads(l) for l in open(run.HERE / "data" / "full.jsonl")]
    sample = random.Random(seed).sample(items, n)
    kept, reasons, cost = [], {}, 0.0

    def one(it):
        try:
            out, c = rewrite(it)
            v, why = validate(it, out)
            return it, v, why, c
        except Exception as e:
            return it, None, f"error:{type(e).__name__}", 0

    with cf.ThreadPoolExecutor(8) as ex:
        for it, v, why, c in ex.map(one, sample):
            cost += c
            if v is None:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                kept.append({**it, "variants": v})
    with open(run.HERE / "data" / "contam.jsonl", "w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")
    for name in ("anon", "cite_only", "no_cite"):
        with open(run.HERE / "data" / f"contam_{name}.jsonl", "w") as f:
            for it in kept:
                f.write(json.dumps({**it, "prompt": it["variants"][name]}) + "\n")
    # original text of the same kept items, as the baseline set
    with open(run.HERE / "data" / "contam_orig.jsonl", "w") as f:
        for it in kept:
            f.write(json.dumps(it) + "\n")
    print(f"kept {len(kept)}/{n}  dropped {reasons}  rewrite cost ${cost:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 500, int(sys.argv[3]) if len(sys.argv) > 3 else 11)
