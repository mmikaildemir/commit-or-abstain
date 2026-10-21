"""Fetch a random sample of CaseHOLD test items through the HF dataset-viewer rows API
(no full download) and build the two conditions.

Usage: python fetch.py [n=1000] [seed=7]      (n=all fetches the full test split -> data/full.jsonl)
Writes data/sample.jsonl with, per item:
  original: 5 options, gold index
  abstain:  same 5 options; in ~30% of items the gold holding is swapped for another real
            holding, chosen as CaseHOLD chose its distractors (most TF-IDF-similar to the
            gold holding, below a 0.75 similarity cap), so the correct answer becomes NONE
"""
import json
import pathlib
import random
import sys
import time
import urllib.request

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API = "https://datasets-server.huggingface.co/rows?dataset=casehold/casehold&config=all&split=test"
TOTAL = 5314
BLOCK = 100
SWAP_RATE = 0.3
SIM_CAP = 0.75  # same upper threshold Zheng et al. (2021) used for distractors
OUT = pathlib.Path(__file__).parent / "data"


def rows(offset, length):
    cache = OUT / "rows_cache" / f"{offset}_{length}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}&offset={offset}&length={length}", timeout=60) as r:
                got = [x["row"] for x in json.load(r)["rows"]]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(got))
            return got
        except Exception:
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"rows {offset}+{length} failed")


def main(n, seed):
    rng = random.Random(seed)
    # blocks of 100 consecutive rows at random offsets (few requests; the API rate-limits)
    picked = []
    offsets = range(0, TOTAL, BLOCK) if n is None else sorted(rng.sample(range(0, TOTAL - BLOCK, BLOCK), n // BLOCK))
    for off in offsets:
        picked += rows(off, min(BLOCK, TOTAL - off))
        time.sleep(2)
    items = [{"id": r["example_id"], "prompt": r["citing_prompt"],
              "options": [r[f"holding_{i}"] for i in range(5)], "gold": int(r["label"])} for r in picked]

    # replacement pool: every "holding ..." statement in the sample, tagged with its source item
    pool = [(j, o) for j, it in enumerate(items) for o in it["options"] if o.lower().startswith("holding")]
    vec = TfidfVectorizer().fit([o for _, o in pool])
    pool_m = vec.transform([o for _, o in pool])

    swapped = sims = 0
    for i, it in enumerate(items):
        opts = list(it["options"])
        gold_text = opts[it["gold"]]
        if rng.random() < SWAP_RATE:
            s = cosine_similarity(vec.transform([gold_text]), pool_m)[0]
            own = set(opts)
            best = max((k for k, (j, o) in enumerate(pool) if j != i and o not in own and s[k] < SIM_CAP),
                       key=lambda k: s[k])
            opts[it["gold"]] = pool[best][1]
            it["abstain"] = {"options": opts, "gold": "NONE", "replaced": gold_text,
                             "replacement_sim": round(float(s[best]), 3)}
            swapped += 1
            sims += s[best]
        else:
            it["abstain"] = {"options": opts, "gold": it["gold"]}
    OUT.mkdir(exist_ok=True)
    with open(OUT / ("full.jsonl" if n is None else "sample.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(f"items={len(items)} swapped={swapped} mean replacement similarity={sims / max(swapped, 1):.3f}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "1000"
    main(None if arg == "all" else int(arg), int(sys.argv[2]) if len(sys.argv) > 2 else 7)
