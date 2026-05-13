"""
Generate static and animated figures for report.md and the LaTeX report.
Run from repo root: python scripts/generate_report_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_assets")
os.makedirs(OUT, exist_ok=True)

# Metrics from BEIR EvaluateRetrieval on SciFact test (committed result files).
ROWS = [
    ("BM25 (sparse)", 0.5597, 0.58389, 0.79294),
    ("Dense (MiniLM + FAISS IP)", 0.64508, 0.67665, 0.925),
    ("Hybrid alpha=0.3", 0.68222, 0.71175, 0.93667),
    ("Hybrid alpha=0.5", 0.67512, 0.70161, 0.93000),
    ("Hybrid alpha=0.7", 0.62439, 0.66390, 0.92533),
    ("RRF k=10", 0.65923, 0.68743, 0.92867),
    ("RRF k=60", 0.63906, 0.67522, 0.92867),
    ("RRF k=100", 0.63354, 0.67364, 0.92867),
]


def bar_comparison():
    labels = [r[0] for r in ROWS]
    ndcg = [r[1] for r in ROWS]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(labels)))
    bars = ax.barh(labels, ndcg, color=colors)
    ax.set_xlabel("NDCG@10")
    ax.set_title("Retrieval quality on SciFact (NDCG@10)")
    ax.set_xlim(0, max(ndcg) * 1.12)
    for b, v in zip(bars, ndcg):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    p = os.path.join(OUT, "ndcg10_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Wrote", p)


def recall_ndcg_scatter():
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, ndcg10, ndcg100, recall100 in ROWS:
        ax.scatter(recall100, ndcg10, s=80, alpha=0.85)
        ax.annotate(name.replace(" ", "\n"), (recall100, ndcg10), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Tradeoff: recall depth vs ranking quality (top 10)")
    plt.tight_layout()
    p = os.path.join(OUT, "recall_vs_ndcg10.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Wrote", p)


def animate_alpha_sweep():
    """Animated bar chart interpolating hybrid alpha emphasis (conceptual sweep)."""
    alphas = np.linspace(0.3, 0.7, 25)
    ndcg_by_alpha = np.interp(
        alphas,
        [0.3, 0.5, 0.7],
        [0.68222, 0.67512, 0.62439],
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    (line,) = ax.plot([], [], "o-", lw=2, markersize=6, color="#2c7fb8")
    ax.set_xlim(0.28, 0.72)
    ax.set_ylim(0.61, 0.70)
    ax.set_xlabel("BM25 weight alpha (dense weight is 1 minus alpha)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Hybrid fusion: alpha vs NDCG@10 (SciFact)")

    def init():
        line.set_data([], [])
        return (line,)

    def frame(i):
        x = alphas[: i + 1]
        y = ndcg_by_alpha[: i + 1]
        line.set_data(x, y)
        return (line,)

    anim = FuncAnimation(fig, frame, frames=len(alphas), init_func=init, blit=True)
    gif_path = os.path.join(OUT, "hybrid_alpha_ndcg10_sweep.gif")
    anim.save(gif_path, writer=PillowWriter(fps=8))
    plt.close(fig)
    print("Wrote", gif_path)


def animate_method_ranking():
    """Cycle through methods highlighting one at a time (moving emphasis)."""
    labels_short = [
        "BM25",
        "Dense",
        "H a0.3",
        "H a0.5",
        "H a0.7",
        "RRF10",
        "RRF60",
        "RRF100",
    ]
    ndcg = [r[1] for r in ROWS]
    fig, ax = plt.subplots(figsize=(8, 4))

    def frame(i):
        ax.clear()
        colors = ["#e34a33" if j == (i % len(ndcg)) else "#bdbdbd" for j in range(len(ndcg))]
        ax.bar(labels_short, ndcg, color=colors)
        ax.set_ylim(0.5, 0.72)
        ax.set_ylabel("NDCG@10")
        ax.set_title("Methods compared (highlight cycles for readability)")
        for j, v in enumerate(ndcg):
            ax.text(j, v + 0.008, f"{v:.2f}", ha="center", fontsize=8)

    anim = FuncAnimation(fig, frame, frames=len(ndcg) * 3, interval=450)
    gif_path = os.path.join(OUT, "methods_highlight_cycle.gif")
    anim.save(gif_path, writer=PillowWriter(fps=3))
    plt.close(fig)
    print("Wrote", gif_path)


if __name__ == "__main__":
    os.chdir(ROOT)
    bar_comparison()
    recall_ndcg_scatter()
    animate_alpha_sweep()
    animate_method_ranking()
    print("Done. Figures in", OUT)
