"""Combination policies on the full split (abstain condition), thresholds tuned on a held-out 30%.

Each item ends as one of:
  right       committed to the correct holding, or said NONE when the holding was absent
  wrong_cite  committed to a holding that is not the one the court cited
  review      said NONE / sent to a person although the correct holding was present
Error cost = k * wrong_cite + review; k (how much worse a wrong citation is than an extra review)
is a policy parameter, so thresholds are tuned on the 30% dev split separately for each k.

Usage: CASEHOLD_SET=full python combine_full.py   -> prints tables, writes combine_full.json
"""
import json
import random

import run

S1 = {"Jev": "typesafe_jev-1.14", "pplx-decider": "perplexity_pplx-decider-v1.2", "d1-ultra": "liquid_d1-ultra"}
LLM = {"Gemini 3.8 Flash": "ans-google_gemini-3.8-flash", "Claude Haiku 5.5": "ans-anthropic_claude-haiku-5.5",
       "GPT-6 Luna": "ans-openai_gpt-6-luna", "GPT-4o-mini": "ans-openai_gpt-4o-mini",
       "Mistral Small": "ans-mistralai_mistral-small-2603", "GLM 5.3 Flash": "ans-z-ai_glm-5.3-flash"}
KS = (1, 5, 10)
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]


def load(f):
    return {r["id"]: r for r in map(json.loads, open(run.LOGS / f"{f}__abstain.jsonl"))}


def outcome(pred, gold):
    if pred == gold:
        return "right"
    return "review" if pred == "NONE" else "wrong_cite"


def score(preds, gold, ids, cost):
    c = {"right": 0, "wrong_cite": 0, "review": 0}
    for i in ids:
        c[outcome(preds[i], gold[i])] += 1
    n = len(ids)
    return {k: v / n for k, v in c.items()} | {"cost_per_1k": 1000 * sum(cost[i] for i in ids) / n}


def policies(s, l, ids):
    """Yield (name, threshold, preds, per-item cost) for every policy and threshold."""
    p_none = {i: (s[i]["probs"] or {}).get("NONE", 0.0) for i in ids}
    p_top = {i: max((s[i]["probs"] or {"x": 0}).values()) for i in ids}
    both = {i: (s[i]["cost"] or 0) + (l[i]["cost"] or 0) for i in ids}
    yield "S1 alone", None, {i: s[i]["pred"] for i in ids}, {i: s[i]["cost"] or 0 for i in ids}
    yield "LLM alone", None, {i: l[i]["pred"] for i in ids}, {i: l[i]["cost"] or 0 for i in ids}
    yield "Agreement gate", None, {i: l[i]["pred"] if l[i]["pred"] == s[i]["pred"] else "NONE" for i in ids}, both
    for t in THRESHOLDS:
        yield "Veto", t, {i: "NONE" if p_none[i] >= t else l[i]["pred"] for i in ids}, both
        # S1 first: S1 answers when its top probability >= t, otherwise the LLM is called and answers
        yield ("S1 first", t, {i: s[i]["pred"] if p_top[i] >= t else l[i]["pred"] for i in ids},
               {i: (s[i]["cost"] or 0) + (0 if p_top[i] >= t else (l[i]["cost"] or 0)) for i in ids})


def main():
    s_logs = {n: load(f) for n, f in S1.items()}
    l_logs = {n: load(f) for n, f in LLM.items()}
    common = set.intersection(*[set(d) for d in list(s_logs.values()) + list(l_logs.values())])
    gold = {i: s_logs["Jev"][i]["gold"] for i in common}
    ids = sorted(common)
    random.Random(2026).shuffle(ids)
    dev, test = ids[:int(0.3 * len(ids))], ids[int(0.3 * len(ids)):]
    print(f"items: {len(ids)} (dev {len(dev)}, test {len(test)})\n")

    results = []
    for sn, s in s_logs.items():
        for ln, l in l_logs.items():
            cands = list(policies(s, l, ids))
            for k in KS:
                best = {}
                for name, t, preds, cost in cands:  # tune each policy's threshold on dev for this k
                    d = score(preds, gold, dev, cost)
                    err = k * d["wrong_cite"] + d["review"]
                    if name not in best or err < best[name][0]:
                        best[name] = (err, t, preds, cost)
                for name, (_, t, preds, cost) in best.items():
                    r = score(preds, gold, test, cost)
                    results.append({"s1": sn, "llm": ln, "k": k, "policy": name, "threshold": t, **r,
                                    "weighted_error": k * r["wrong_cite"] + r["review"]})
    json.dump(results, open(run.HERE / "combine_full.json", "w"), indent=1)

    for k in KS:
        print(f"=== k = {k}: a wrong citation costs {k}x an unnecessary review (test split, thresholds from dev)")
        print(f"{'S1':13s} {'LLM':17s} {'policy':15s} {'t':>5s} {'right':>6s} {'wrong':>6s} {'review':>6s} {'w.err':>6s} {'$/1k':>7s}")
        for sn in S1:
            for ln in LLM:
                rows = [r for r in results if r["s1"] == sn and r["llm"] == ln and r["k"] == k]
                for r in sorted(rows, key=lambda r: r["weighted_error"]):
                    t = "" if r["threshold"] is None else f"{r['threshold']:.2f}"
                    print(f"{sn:13s} {ln:17s} {r['policy']:15s} {t:>5s} {r['right']:6.3f} {r['wrong_cite']:6.3f} "
                          f"{r['review']:6.3f} {r['weighted_error']:6.3f} {r['cost_per_1k']:7.3f}")
            print()


if __name__ == "__main__":
    main()
