import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def mix_gauss(x, comps):
    """Return a normalized mixture of Gaussians over x."""
    y = np.zeros_like(x, dtype=float)
    for weight, mean, std in comps:
        y += weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return y


def gaussian_kde_1d(x, samples, bandwidth):
    """Simple Gaussian KDE without external deps."""
    diffs = (x[:, None] - samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs**2) / (bandwidth * np.sqrt(2 * np.pi))
    return kernel.mean(axis=1)


def plot_projection_schematic():
    x = np.linspace(-4, 4, 1200)

    # Gray: pre-train/base policy projection
    p0 = mix_gauss(x, [(0.55, -1.8, 0.9), (0.45, 1.2, 1.1)])

    # Blue: SFT-style sharpening (moderate)
    psft = mix_gauss(x, [(0.45, -1.8, 1.0), (0.55, 1.5, 0.45)])

    # Blue: Off-policy DPO/IPO style (sharper peak, squeezed middle)
    pdpo = mix_gauss(x, [(0.85, -2.1, 0.22), (0.15, 2.4, 0.40)])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    # Panel 1: SFT
    ax = axes[0]
    ax.plot(x, p0, lw=3, alpha=0.45, color="gray", label="pre")
    ax.plot(x, psft, lw=3, color="tab:blue", label="after")
    ax.set_title("SFT")
    ax.set_yticks([])
    ax.set_xticks([])

    # Mark y_u^+ with a red arrow
    yu_plus = 1.5
    yu_idx = np.argmin(np.abs(x - yu_plus))
    ax.annotate(
        "",
        xy=(yu_plus, psft[yu_idx] * 1.15),
        xytext=(yu_plus, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="tab:red"),
    )
    ax.text(yu_plus + 0.05, 0.02, r"$y_u^+$")

    # Panel 2: Off-policy DPO/IPO
    ax = axes[1]
    ax.plot(x, p0, lw=3, alpha=0.45, color="gray", label="pre")
    ax.plot(x, pdpo, lw=3, color="tab:blue", label="after")
    ax.set_title("Off-policy DPO / IPO")
    ax.set_yticks([])
    ax.set_xticks([])

    # Mark y* with a dashed line
    y_star = -2.1
    ax.axvline(y_star, ls="--", lw=2, color="black")
    ax.text(y_star - 0.05, 0.02, r"$y^*$")

    # Squeezing region annotation
    ax.annotate(
        "squeezing",
        xy=(0.2, 0.25),
        xytext=(0.2, 0.55),
        arrowprops=dict(arrowstyle="<->", lw=2),
        ha="center",
    )

    fig.suptitle("Projected Distribution Shift", fontsize=12, y=1.03)
    fig.tight_layout()


def plot_discrete_candidates(data_dir):
    df = pd.read_csv(data_dir / "discrete_candidates.csv")
    quality = df["quality"].to_numpy()
    logp0 = df["logp_pre"].to_numpy()
    logpt = df["logp_post"].to_numpy()

    def softmax(x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

    p0 = softmax(logp0)
    pt = softmax(logpt)

    order = np.argsort(quality)
    p0 = p0[order]
    pt = pt[order]

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    idx = np.arange(len(p0))
    ax.vlines(idx - 0.12, 0, p0, color="gray", alpha=0.55, linewidth=2.2)
    ax.vlines(idx + 0.12, 0, pt, color="tab:blue", linewidth=2.2)
    ax.set_title("Discrete Candidate Policy (same responses)")
    ax.set_xlabel("Candidate response rank (sorted by quality)")
    ax.set_ylabel("Probability mass")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(["pre", "after"], frameon=False)
    fig.tight_layout()


def plot_semantic_projection(data_dir):
    df = pd.read_csv(data_dir / "semantic_projection.csv")
    pre = df["z_pre"].to_numpy()
    post = df["z_post"].to_numpy()

    x = np.linspace(-4, 4, 800)
    kde_pre = gaussian_kde_1d(x, pre, bandwidth=0.35)
    kde_post = gaussian_kde_1d(x, post, bandwidth=0.3)

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.plot(x, kde_pre, color="gray", lw=3, alpha=0.5, label="pre")
    ax.plot(x, kde_post, color="tab:blue", lw=3, label="after")
    ax.set_title("Semantic Projection KDE (1D)")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(frameon=False)
    fig.tight_layout()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data" / "policy_projection_demo"

    plot_projection_schematic()
    plot_discrete_candidates(data_dir)
    plot_semantic_projection(data_dir)
    plt.show()
