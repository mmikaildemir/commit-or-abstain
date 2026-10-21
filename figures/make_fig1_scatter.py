"""Figure 1(b): wrong-holding rate when the correct holding is absent vs. cost per 1,000 queries.

Reads fig1_data.json (written from casehold/logs_full) and writes fig1_scatter.pdf (+ .png preview).
Palette: reference categorical slots 1-2 (validated all-pairs, light mode); shape is the
secondary encoding so the figure survives grayscale print.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

HERE = pathlib.Path(__file__).parent
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"

# label offsets in points (dx, dy) and alignment, hand-placed to avoid collisions
LABELS = {
    "Jev": ((7, -3), "left"),
    "pplx-decider": ((7, 0), "left"),
    "d1-ultra": ((-7, -3), "right"),
    "Luna Decisions": ((5, -8), "left"),
    "GLM 5.3 Flash": ((-6, 9), "right"),
    "Claude Haiku 5.5": ((-7, -2), "right"),
    "GPT-4o-mini": ((7, 2), "left"),
    "GPT-6 Luna": ((7, 0), "left"),
    "Mistral Small": ((7, -2), "left"),
    "Gemini 3.8 Flash": ((0, -11), "center"),
}

LEADERS = {"GLM 5.3 Flash"}  # labels whose nearest dot would otherwise be ambiguous

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "pdf.fonttype": 42})

data = json.loads((HERE / "fig1_data.json").read_text())
fig, ax = plt.subplots(figsize=(2.95, 2.5))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for kind, color, marker, label in (("S1", BLUE, "^", "System One model"), ("LLM", ORANGE, "o", "Generative LLM")):
    pts = [d for d in data if d["kind"] == kind]
    ax.scatter([d["cost_per_1k"] for d in pts], [100 * d["wrong_when_absent"] for d in pts], s=34 if marker == "o" else 40,
               marker=marker, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=label)

for d in data:
    (dx, dy), ha = LABELS[d["name"]]
    leader = dict(arrowstyle="-", color=INK2, lw=0.5, shrinkA=1, shrinkB=4) if d["name"] in LEADERS else None
    ax.annotate(d["name"], (d["cost_per_1k"], 100 * d["wrong_when_absent"]), xytext=(dx, dy),
                textcoords="offset points", ha=ha, va="center", fontsize=6.6, color=INK, arrowprops=leader)

ax.set_xscale("log")
ax.set_xlim(0.008, 1.5)
ax.set_ylim(20, 95)
ax.xaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3, 1]))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Cost per 1,000 questions (USD, log scale)")
ax.set_ylabel("Picks a holding when none is correct")
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_linewidth(0.6)
ax.spines["bottom"].set_linewidth(0.6)
ax.tick_params(length=2, width=0.6)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=6.6, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)

fig.tight_layout(pad=0.3)
fig.savefig(HERE / "fig1_scatter.pdf")
fig.savefig(HERE / "fig1_scatter.png", dpi=300)
print("wrote fig1_scatter.pdf/.png")
