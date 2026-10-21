"""Final numbers for the paper: per-model table with bootstrap CIs, macro F1, calibration, cost.

Usage: CASEHOLD_SET=full python final_stats.py   -> prints LaTeX-ready rows, writes final_stats.json
"""
import json
import random

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

import run

MODELS = [  # (display name, log stem, kind)
    ("Jev 1.14", "typesafe_jev-1.14", "S1"), ("pplx-decider v1.2", "perplexity_pplx-decider-v1.2", "S1"),
    ("d1-ultra", "liquid_d1-ultra", "S1"), ("GPT-6 Luna Decisions", "openai_gpt-6-luna-decisions", "S1"),
    ("Gemini 3.8 Flash", "ans-google_gemini-3.8-flash", "LLM"), ("Claude Haiku 5.5", "ans-anthropic_claude-haiku-5.5", "LLM"),
    ("GPT-6 Luna", "ans-openai_gpt-6-luna", "LLM"), ("Mistral Small", "ans-mistralai_mistral-small-2603", "LLM"),
    ("GLM 5.3 Flash", "ans-z-ai_glm-5.3-flash", "LLM"), ("GPT-4o-mini", "ans-openai_gpt-4o-mini", "LLM"),
]
B = 2000


def load(stem, cond):
    return [json.loads(l) for l in open(run.LOGS / f"{stem}__{cond}.jsonl")]


def boot_ci(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def ece(rows, b=10):
    c = np.array([r["confidence"] for r in rows if r["confidence"] is not None])
    a = np.array([r["pred"] == r["gold"] for r in rows if r["confidence"] is not None])
    if len(c) == 0:
        return None
    t = 0.0
    for i in range(b):
        m = (c >= i / b) & ((c < (i + 1) / b) | (i == b - 1))
        if m.any():
            t += m.mean() * abs(a[m].mean() - c[m].mean())
    return t


def main():
    rng = np.random.default_rng(2026)
    out = []
    for name, stem, kind in MODELS:
        row = {"name": name, "kind": kind}
        for cond in ("original", "abstain"):
            rows = load(stem, cond)
            correct = [r["pred"] == r["gold"] for r in rows]
            labels = sorted({r["gold"] for r in rows})
            row[f"{cond}_n"] = len(rows)
            row[f"{cond}_acc"] = float(np.mean(correct))
            row[f"{cond}_acc_ci"] = boot_ci(correct, rng)
            row[f"{cond}_macro_f1"] = float(f1_score([r["gold"] for r in rows], [r["pred"] or "X" for r in rows],
                                                     labels=labels, average="macro"))
            row[f"{cond}_ece"] = ece(rows)
            row[f"{cond}_cost_per_1k"] = 1000 * sum(r["cost"] or 0 for r in rows) / len(rows)
            if cond == "abstain":
                absent = [r for r in rows if r["gold"] == "NONE"]
                present = [r for r in rows if r["gold"] != "NONE"]
                wrong = [r["pred"] != "NONE" for r in absent]
                false_none = [r["pred"] == "NONE" for r in present]
                row["wrong_when_absent"] = float(np.mean(wrong))
                row["wrong_when_absent_ci"] = boot_ci(wrong, rng)
                row["false_none"] = float(np.mean(false_none))
                row["false_none_ci"] = boot_ci(false_none, rng)
                if all(r["probs"] and "NONE" in r["probs"] for r in rows):
                    row["auroc_none"] = float(roc_auc_score([r["gold"] == "NONE" for r in rows],
                                                            [r["probs"]["NONE"] for r in rows]))
        out.append(row)
        f = lambda v, ci: f"{100 * v:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
        print(f"{name:22s} orig {f(row['original_acc'], row['original_acc_ci'])} mF1 {row['original_macro_f1']:.3f} | "
              f"abst {f(row['abstain_acc'], row['abstain_acc_ci'])} | wrong|absent {f(row['wrong_when_absent'], row['wrong_when_absent_ci'])} | "
              f"falseNONE {f(row['false_none'], row['false_none_ci'])} | ECE o/a {row['original_ece'] if row['original_ece'] is None else round(row['original_ece'], 3)}/"
              f"{row['abstain_ece'] if row['abstain_ece'] is None else round(row['abstain_ece'], 3)} | AUROC {row.get('auroc_none', '-') if 'auroc_none' not in row else round(row['auroc_none'], 3)} | "
              f"$/1k o/a {row['original_cost_per_1k']:.3f}/{row['abstain_cost_per_1k']:.3f}")
    json.dump(out, open(run.HERE / "final_stats.json", "w"), indent=1)

    # paired comparisons for headline claims (McNemar exact via binomial on discordant pairs)
    from scipy.stats import binomtest
    def paired(a_stem, b_stem, cond, metric):
        A = {r["id"]: r for r in load(a_stem, cond)}
        Bm = {r["id"]: r for r in load(b_stem, cond)}
        ids = [i for i in A if i in Bm and metric(A[i]) is not None]
        a = np.array([metric(A[i]) for i in ids], float)
        b = np.array([metric(Bm[i]) for i in ids], float)
        d = a - b
        idx = rng.integers(0, len(d), (B, len(d)))
        ci = np.percentile(d[idx].mean(1), [2.5, 97.5])
        n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        return d.mean(), ci, p, len(ids)
    acc = lambda r: float(r["pred"] == r["gold"])
    wrong_abs = lambda r: float(r["pred"] != "NONE") if r["gold"] == "NONE" else None
    print("\nPaired comparisons (difference A-B, 95% bootstrap CI, exact McNemar p):")
    for a, b, cond, m, lab in [
        ("perplexity_pplx-decider-v1.2", "ans-google_gemini-3.8-flash", "original", acc, "accuracy"),
        ("typesafe_jev-1.14", "ans-z-ai_glm-5.3-flash", "original", acc, "accuracy"),
        ("typesafe_jev-1.14", "ans-google_gemini-3.8-flash", "abstain", wrong_abs, "wrong-when-absent"),
        ("ans-openai_gpt-4o-mini", "ans-google_gemini-3.8-flash", "abstain", wrong_abs, "wrong-when-absent"),
        ("typesafe_jev-1.14", "ans-z-ai_glm-5.3-flash", "abstain", wrong_abs, "wrong-when-absent"),
    ]:
        d, ci, p, n = paired(a, b, cond, m)
        print(f"  {a} - {b} [{cond} {lab}] {100 * d:+.1f} pts [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}] p={p:.2g} n={n}")


if __name__ == "__main__":
    main()
