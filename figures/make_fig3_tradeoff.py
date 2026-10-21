"""Figure 2: wrong citations vs. items sent to review on the abstain condition (70% test split).

Points: each model alone. Curves: veto policy (language model answers unless the decision model's
P(none) >= t) across thresholds. Diamonds: agreement gate. Reads casehold/logs_full directly.
"""
import json
import pathlib
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = pathlib.Path(__file__).parent
LOGS = HERE.parent / "logs_full"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

S1 = {"Jev": "typesafe_jev-1.14", "pplx-decider": "perplexity_pplx-decider-v1.2", "d1-ultra": "liquid_d1-ultra",
      "Luna Decisions": "openai_gpt-6-luna-decisions"}
LLM = {"Gemini 3.8 Flash": "ans-google_gemini-3.8-flash", "Claude Haiku 5.5": "ans-anthropic_claude-haiku-5.5",
       "GPT-6 Luna": "ans-openai_gpt-6-luna", "Mistral Small": "ans-mistralai_mistral-small-2603",
       "GLM 5.3 Flash": "ans-z-ai_glm-5.3-flash", "GPT-4o-mini": "ans-openai_gpt-4o-mini"}


def load(stem):
    return {r["id"]: r for r in map(json.loads, open(LOGS / f"{stem}__abstain.jsonl"))}


logs = {n: load(s) for n, s in {**S1, **LLM}.items()}
ids = sorted(set.intersection(*[set(d) for d in logs.values()]))
random.Random(2026).shuffle(ids)            # same split as combine_full.py
test = ids[int(0.3 * len(ids)):]
gold = {i: logs["Jev"][i]["gold"] for i in test}


def rates(pred):
    wrong = sum(pred[i] != gold[i] and pred[i] != "NONE" for i in test) / len(test)
    review = sum(pred[i] == "NONE" and gold[i] != "NONE" for i in test) / len(test)
    return 100 * review, 100 * wrong


alone = {n: rates({i: d[i]["pred"] for i in test}) for n, d in logs.items()}


def veto(s, l, t):
    return rates({i: "NONE" if (logs[s][i]["probs"] or {}).get("NONE", 0) >= t else logs[l][i]["pred"] for i in test})


def gate(s, l):
    return rates({i: logs[l][i]["pred"] if logs[l][i]["pred"] == logs[s][i]["pred"] else "NONE" for i in test})


TS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
curves = {"d1-ultra veto + Gemini 3.8 Flash": [veto("d1-ultra", "Gemini 3.8 Flash", t) for t in TS],
          "pplx-decider veto + GLM 5.3 Flash": [veto("pplx-decider", "GLM 5.3 Flash", t) for t in TS]}
gates = {"pplx-decider + Gemini 3.8 Flash": gate("pplx-decider", "Gemini 3.8 Flash"),
         "pplx-decider + GLM 5.3 Flash": gate("pplx-decider", "GLM 5.3 Flash")}
json.dump({"n_test": len(test), "alone": alone, "curves": curves, "thresholds": TS, "gates": gates},
          open(HERE / "fig2_data.json", "w"), indent=1)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.0, 2.9))
for (name, pts), ls in zip(curves.items(), ("-", "--")):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=AQUA, lw=1.6, zorder=2, label=f"Veto: {name.replace(' veto', '')}")
for name, (x, y) in gates.items():
    ax.scatter([x], [y], marker="D", s=30, color=AQUA, edgecolor=SURFACE, linewidth=1.0, zorder=4)
for n, (x, y) in alone.items():
    ax.scatter([x], [y], marker="^" if n in S1 else "o", s=36, color=BLUE if n in S1 else ORANGE,
               edgecolor=SURFACE, linewidth=1.0, zorder=3)

LAB = {  # (dx, dy, ha) in points, or ("at", x, y) = label placed at data coords with a leader line
    "Jev": (6, 0, "left"), "pplx-decider": ("at", -0.6, 28.5), "d1-ultra": (6, 0, "left"),
    "Claude Haiku 5.5": (6, 1, "left"), "GPT-4o-mini": (6, 1, "left"),
    "Mistral Small": ("at", 15, 40.5), "GPT-6 Luna": ("at", 15, 37.5),
    "Luna Decisions": ("at", 15, 34.5), "GLM 5.3 Flash": ("at", 15, 31.5),
    "Gemini 3.8 Flash": ("at", 0.2, 22.5),
}
for n, (x, y) in alone.items():
    spec = LAB[n]
    if spec[0] == "at":
        ax.annotate(n, (x, y), xytext=(spec[1], spec[2]), textcoords="data", ha="left", va="center", fontsize=6.4,
                    color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=3))
    else:
        dx, dy, ha = spec
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", fontsize=6.4, color=INK)
for name, (x, y) in gates.items():
    short = "Agreement: pplx + " + ("Gemini" if "Gemini" in name else "GLM")
    left = "Gemini" in name
    ax.annotate(short, (x, y), xytext=(0, -9) if left else (6, 0), textcoords="offset points", ha="center" if left else "left", va="center",
                fontsize=6.4, color=INK)

ax.set_xlabel("Items sent to review although the correct holding was present (%)")
ax.set_ylabel("Wrong citations (%)")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlim(-1, 62)
ax.set_ylim(0, 52)
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(0.6)
ax.tick_params(length=2, width=0.6)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="^", ls="", color=BLUE, label="System One model alone"),
           Line2D([], [], marker="o", ls="", color=ORANGE, label="Generative LLM alone"),
           Line2D([], [], ls="-", color=AQUA, label="Veto: d1-ultra + Gemini 3.8 Flash"),
           Line2D([], [], ls="--", color=AQUA, label="Veto: pplx-decider + GLM 5.3 Flash"),
           Line2D([], [], marker="D", ls="", color=AQUA, label="Agreement gate")]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.4, handletextpad=0.4)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig2_tradeoff.pdf")
fig.savefig(HERE / "fig2_tradeoff.png", dpi=300)
print("wrote fig2_tradeoff.pdf/.png; test n =", len(test))
for k, v in gates.items():
    print(k, [round(x, 1) for x in v])
