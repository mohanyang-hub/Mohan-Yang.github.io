from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(r"C:\Users\Che\Documents\Codex\2026-05-21\new-chat-2")
INPUT_DIR = ROOT / "amaf_reasonable_attention_lam030_outputs"
INPUT_CSV = INPUT_DIR / "attention_by_grade_AMAFNet_ReasonableAttention_Lam030_Epoch60.csv"
OUT_DIR = INPUT_DIR / "advanced_attention_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRADES = ["G1", "G3", "G5", "G7", "G9"]
MODALITIES = ["Spectral-SPAD", "Handcrafted", "R-G-NIR", "3D Structure"]

COLORS = {
    "Spectral-SPAD": "#2B6CB0",
    "Handcrafted": "#D18B22",
    "R-G-NIR": "#2E9D73",
    "3D Structure": "#8E3B63",
}


def set_journal_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "axes.unicode_minus": False,
            "axes.linewidth": 1.8,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
        }
    )


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.tick_params(axis="both", labelsize=15, width=1.6, length=5.5)
    ax.grid(False)


def load_attention():
    df = pd.read_csv(INPUT_CSV)
    return df.set_index("Grade").loc[GRADES, MODALITIES].astype(float)


def plot_journal_attention():
    set_journal_style()
    df = load_attention()
    x = np.arange(len(GRADES))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), dpi=220)
    ax1, ax2 = axes

    for modality in MODALITIES:
        ax1.plot(
            x,
            df[modality].to_numpy(),
            color=COLORS[modality],
            linewidth=3.0,
            marker="o",
            markersize=8.8,
            markeredgecolor="white",
            markeredgewidth=1.4,
            label=modality,
        )

    ax1.set_title("(a)", fontsize=17, fontweight="bold", loc="left", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(GRADES, fontsize=16)
    ax1.set_xlabel("")
    ax1.set_ylabel("Attention weight", fontsize=18)
    ax1.set_ylim(0.10, 0.39)
    ax1.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ax1.set_xlim(-0.25, len(x) - 0.75)
    style_axis(ax1)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[m],
            linewidth=3.0,
            marker="o",
            markersize=8.5,
            markeredgecolor="white",
            label=m,
        )
        for m in MODALITIES
    ]
    ax1.legend(
        handles=handles,
        frameon=False,
        fontsize=13.2,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.00, 0.01),
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    ranks = df.rank(axis=1, ascending=False, method="first")
    for modality in MODALITIES:
        y_rank = ranks[modality].to_numpy()
        ax2.plot(
            x,
            y_rank,
            color=COLORS[modality],
            linewidth=3.0,
            marker="o",
            markersize=21.5,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )
        for xi, yi, val in zip(x, y_rank, df[modality].to_numpy()):
            ax2.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=11.2, color="white")

    ax2.set_title("(b)", fontsize=17, fontweight="bold", loc="left", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(GRADES, fontsize=16)
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(["1st", "2nd", "3rd", "4th"], fontsize=15)
    ax2.invert_yaxis()
    ax2.set_xlim(-0.25, len(x) - 0.75)
    ax2.set_ylim(4.45, 0.55)
    ax2.set_ylabel("Modality rank", fontsize=18)
    style_axis(ax2)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.17, wspace=0.24)
    png = OUT_DIR / "Fig_AMAF_attention_journal_style.png"
    pdf = OUT_DIR / "Fig_AMAF_attention_journal_style.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


if __name__ == "__main__":
    png_path, pdf_path = plot_journal_attention()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
