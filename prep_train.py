"""Convert a CaseHOLD parquet split to our jsonl item format, with the abstain condition built the
same way as fetch.py (30% of items: gold holding swapped for the most TF-IDF-similar holding from
the split's own pool, below the 0.75 cap).

Usage: python prep_train.py data/train.parquet data/train.jsonl [seed=7]
Needs pandas + pyarrow + scikit-learn (run on the lab server's casehold-env).
"""
import json
import random
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SWAP_RATE = 0.3
SIM_CAP = 0.75


def main(src, dst, seed=7):
    df = pd.read_parquet(src)
    items = [{"id": int(r.example_id), "prompt": r.citing_prompt,
              "options": [getattr(r, f"holding_{i}") for i in range(5)], "gold": int(r.label)}
             for r in df.itertuples()]
    rng = random.Random(seed)
    swap = [i for i in range(len(items)) if rng.random() < SWAP_RATE]

    pool = [(j, o) for j, it in enumerate(items) for o in it["options"] if o.lower().startswith("holding")]
    vec = TfidfVectorizer().fit([o for _, o in pool])
    pool_m = vec.transform([o for _, o in pool]).T.tocsc()
    owner = np.array([j for j, _ in pool])
    for it in items:
        it["abstain"] = {"options": it["options"], "gold": it["gold"]}
    for start in range(0, len(swap), 2000):  # chunked sparse similarity
        chunk = swap[start:start + 2000]
        q = vec.transform([items[i]["options"][items[i]["gold"]] for i in chunk])
        sims = (q @ pool_m).toarray()
        for row, i in enumerate(chunk):
            s = sims[row]
            s[(owner == i) | (s >= SIM_CAP)] = -1
            own = set(items[i]["options"])
            for k in np.argsort(-s):
                if pool[k][1] not in own:
                    break
            opts = list(items[i]["options"])
            gold_text = opts[items[i]["gold"]]
            opts[items[i]["gold"]] = pool[k][1]
            items[i]["abstain"] = {"options": opts, "gold": "NONE", "replaced": gold_text,
                                   "replacement_sim": round(float(s[k]), 3)}
    with open(dst, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    sw = [it["abstain"]["replacement_sim"] for it in items if it["abstain"]["gold"] == "NONE"]
    print(f"items={len(items)} swapped={len(sw)} mean replacement similarity={np.mean(sw):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 7)
