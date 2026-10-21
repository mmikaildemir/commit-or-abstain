"""Summarize logs/: accuracy, calibration, abstention behaviour, and a Jev -> LLM escalation curve.

Usage: python analyze.py [cheap_model] [strong_model]
  defaults: typesafe/jev-1.14  chat:openai/gpt-6-luna
"""
import json
import os
import math
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).parent
SET = os.environ.get("CASEHOLD_SET", "pilot")  # pilot: data/sample.jsonl + logs/; full: data/full.jsonl + logs_full/
LOGS = HERE / ("logs" if SET == "pilot" else f"logs_{SET}")


def load(model, cond):
    f = LOGS / f"{model.replace('/', '_').replace(':', '-')}__{cond}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(f))} if f.exists() else {}


def conf(r):
    """Probability the model assigns to its own answer."""
    if r.get("probs") and r["pred"] in r["probs"]:
        return r["probs"][r["pred"]]
    return r["confidence"] if r["confidence"] is not None else 0.0


def ece(rows, bins=10):
    tot, n = 0.0, len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if lo <= conf(r) < hi or (b == bins - 1 and conf(r) == 1.0)]
        if sel:
            acc = sum(r["pred"] == r["gold"] for r in sel) / len(sel)
            avg = sum(conf(r) for r in sel) / len(sel)
            tot += len(sel) / n * abs(acc - avg)
    return tot


def selective(rows, coverage):
    k = max(1, round(len(rows) * coverage))
    top = sorted(rows, key=conf, reverse=True)[:k]
    return sum(r["pred"] == r["gold"] for r in top) / k


def paired(a_model, b_model, cond, metric, keep=lambda r: True, iters=5000):
    """Bootstrap 95% CI of metric(a) - metric(b) over shared items passing keep; metric maps a row to 0/1."""
    a, b = load(a_model, cond), load(b_model, cond)
    ids = sorted(i for i in set(a) & set(b) if keep(a[i]))
    if not ids:
        return None
    diffs = [metric(a[i]) - metric(b[i]) for i in ids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in ids) / len(ids) for _ in range(iters))
    return sum(diffs) / len(ids), boots[int(.025 * iters)], boots[int(.975 * iters)]


def summarize(model, cond):
    rows = list(load(model, cond).values())
    if not rows:
        return
    n = len(rows)
    acc = sum(r["pred"] == r["gold"] for r in rows) / n
    half = 1.96 * math.sqrt(acc * (1 - acc) / n)
    cost = sum(r["cost"] or 0 for r in rows)
    out = (f"{model:32s} {cond:8s} n={n:4d} acc={acc:.3f}±{half:.3f} ECE={ece(rows):.3f} "
           f"sel@50%={selective(rows, .5):.3f} sel@80%={selective(rows, .8):.3f} cost=${cost:.4f}")
    if cond == "abstain":
        absent = [r for r in rows if r["gold"] == "NONE"]
        present = [r for r in rows if r["gold"] != "NONE"]
        wrong_commit = [r for r in absent if r["pred"] != "NONE"]
        false_abst = [r for r in present if r["pred"] == "NONE"]
        mc = sum(conf(r) for r in wrong_commit) / len(wrong_commit) if wrong_commit else float("nan")
        out += (f"\n{'':42s} answer absent: abstained {len(absent) - len(wrong_commit)}/{len(absent)}, "
                f"picked a wrong holding {len(wrong_commit)} (mean conf {mc:.2f})"
                f"\n{'':42s} answer present: acc {sum(r['pred'] == r['gold'] for r in present) / len(present):.3f}, "
                f"false abstain {len(false_abst)}/{len(present)}")
    print(out)


def escalation(cheap, strong, cond):
    c, s = load(cheap, cond), load(strong, cond)
    ids = sorted(set(c) & set(s))
    if not ids:
        return
    s_acc = sum(s[i]["pred"] == s[i]["gold"] for i in ids) / len(ids)
    s_cost = sum(s[i]["cost"] or 0 for i in ids)
    print(f"\nEscalation [{cond}] {cheap} -> {strong} (n={len(ids)}; strong alone acc={s_acc:.3f}, cost=${s_cost:.4f})")
    print("  threshold  handled_by_cheap  accuracy  cost      cost_vs_strong")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
        correct, cost, kept = 0, 0.0, 0
        for i in ids:
            cost += c[i]["cost"] or 0
            if conf(c[i]) >= t:
                kept += 1
                correct += c[i]["pred"] == c[i]["gold"]
            else:
                cost += s[i]["cost"] or 0
                correct += s[i]["pred"] == s[i]["gold"]
        print(f"  {t:9.2f}  {kept / len(ids):16.0%}  {correct / len(ids):8.3f}  ${cost:.4f}  {cost / s_cost if s_cost else 0:6.0%}")


if __name__ == "__main__":
    cheap = sys.argv[1] if len(sys.argv) > 1 else "typesafe/jev-1.14"
    strong = sys.argv[2] if len(sys.argv) > 2 else "chat:openai/gpt-6-luna"
    models = sorted({f.stem.split("__")[0] for f in LOGS.glob("*.jsonl")})
    names = {m.replace('/', '_').replace(':', '-'): m for m in
             ["typesafe/jev-1.14", "openai/gpt-6-luna-decisions", "chat:openai/gpt-6-luna", cheap, strong]}
    for cond in ("original", "abstain"):
        for m in models:
            summarize(names.get(m, m), cond)
        print()
    print("Paired differences vs Jev (mean, 95% bootstrap CI)")
    correct = lambda r: r["pred"] == r["gold"]
    for other in ("openai/gpt-6-luna-decisions", "chat:openai/gpt-6-luna"):
        for cond in ("original", "abstain"):
            d = paired("typesafe/jev-1.14", other, cond, correct)
            if d:
                print(f"  accuracy  Jev - {other:30s} {cond:8s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        abst = lambda r: r["pred"] == "NONE"
        for label, keep in (("abstain rate, answer absent ", lambda r: r["gold"] == "NONE"),
                            ("abstain rate, answer present", lambda r: r["gold"] != "NONE")):
            d = paired("typesafe/jev-1.14", other, "abstain", abst, keep)
            if d:
                print(f"  {label}  Jev - {other:30s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
    for cond in ("original", "abstain"):
        escalation(cheap, strong, cond)
