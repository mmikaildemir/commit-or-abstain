"""Figure 3: ROC of each decision model's P(none) on the abstain condition (all 5,314 items).

Curves: decision models, threshold on P(none). Triangles: each decision model's own answer (argmax).
Circles: generative LLMs, which give one answer and so one operating point. Reads casehold/logs_full directly.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = pathlib.Path(__file__).parent
LOGS = HERE.parent / "logs_full"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

S1 = {"pplx-decider": "perplexity_pplx-decider-v1.2", "d1-ultra": "liquid_d1-ultra", "Jev": "typesafe_jev-1.14",
      "Luna Decisions": "openai_gpt-6-luna-decisions"}
LLM = {"Gemini 3.8 Flash": "ans-google_gemini-3.8-flash", "Claude Haiku 5.5": "ans-anthropic_claude-haiku-5.5",
       "GPT-6 Luna": "ans-openai_gpt-6-luna", "Mistral Small": "ans-mistralai_mistral-small-2603",
       "GLM 5.3 Flash": "ans-z-ai_glm-5.3-flash", "GPT-4o-mini": "ans-openai_gpt-4o-mini"}
STYLE = {"pplx-decider": "-", "d1-ultra": "--", "Jev": (0, (1, 1.2)), "Luna Decisions": "-."}


def load(stem):
    return [json.loads(l) for l in open(LOGS / f"{stem}__abstain.jsonl")]


def roc(rows):
    """Points (FPR, TPR) for 'abstain if P(none) >= t' over all thresholds; positive = no correct holding."""
    pos = sum(r["gold"] == "NONE" for r in rows)
    neg = len(rows) - pos
    scored = sorted(((r["probs"] or {}).get("NONE", 0.0), r["gold"] == "NONE") for r in rows)[::-1]
    pts, tp, fp, i = [(0.0, 0.0)], 0, 0, 0
    while i < len(scored):
        t = scored[i][0]
        while i < len(scored) and scored[i][0] == t:
            tp += scored[i][1]
            fp += not scored[i][1]
            i += 1
        pts.append((fp / neg, tp / pos))
    auc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    return pts, auc


def point(rows):
    pos = [r for r in rows if r["gold"] == "NONE"]
    neg = [r for r in rows if r["gold"] != "NONE"]
    return (sum(r["pred"] == "NONE" for r in neg) / len(neg), sum(r["pred"] == "NONE" for r in pos) / len(pos))


curves, own, llm = {}, {}, {}
for n, s in S1.items():
    rows = load(s)
    curves[n] = roc(rows)
    own[n] = point(rows)
for n, s in LLM.items():
    llm[n] = point(load(s))
json.dump({"auroc": {n: a for n, (_, a) in curves.items()}, "own": own, "llm": llm},
          open(HERE / "fig3_data.json", "w"), indent=1)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, zorder=1)
for n, (pts, auc) in curves.items():
    ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=STYLE[n], color=BLUE, lw=1.5, zorder=2)
for n, (x, y) in own.items():
    ax.scatter([x], [y], marker="^", s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.0, zorder=4)
for n, (x, y) in llm.items():
    ax.scatter([x], [y], marker="o", s=30, color=ORANGE, edgecolor=SURFACE, linewidth=1.0, zorder=3)

# curve labels at the right edge of the plot, at each curve's height there
XR = 0.5
NUDGE = {"pplx-decider": 0.025, "d1-ultra": -0.02, "Jev": 0.0, "Luna Decisions": 0.0}
for n, (pts, auc) in curves.items():
    y = max(p[1] for p in pts if p[0] <= XR)
    ax.text(XR + 0.008, y + NUDGE[n], f"{n} ({auc:.2f})", ha="left", va="center", fontsize=6.4, color=INK, clip_on=False)
gx, gy = llm["Gemini 3.8 Flash"]
ax.annotate("Gemini 3.8 Flash", (gx, gy), xytext=(0.01, 0.75), textcoords="data", ha="left", va="center", fontsize=6.4,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=3))

ax.set_xlabel("False abstention: items with a correct holding answered as none")
ax.set_ylabel("Items without a correct holding answered as none")
pct = FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
ax.xaxis.set_major_formatter(pct)
ax.yaxis.set_major_formatter(pct)
ax.set_xlim(-0.005, XR)
ax.set_ylim(0, 1.01)
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(0.6)
ax.tick_params(length=2, width=0.6)
handles = [Line2D([], [], ls="-", color=BLUE, label="System One model, threshold on P(none)"),
           Line2D([], [], marker="^", ls="", color=BLUE, label="System One model, own answer"),
           Line2D([], [], marker="o", ls="", color=ORANGE, label="Generative LLM")]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.04), frameon=False, fontsize=6.4, handletextpad=0.4)
fig.tight_layout(pad=0.3, rect=(0, 0, 0.84, 1))
fig.savefig(HERE / "fig3_roc.pdf")
fig.savefig(HERE / "fig3_roc.png", dpi=300)
print("wrote fig3_roc.pdf/.png")
print({n: round(a, 3) for n, (_, a) in curves.items()})
print({n: tuple(round(v, 3) for v in p) for n, p in llm.items()})
