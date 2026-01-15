import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def mix_gauss(x, comps):
    """返回一维高斯混合密度。"""
    y = np.zeros_like(x, dtype=float)
    for weight, mean, std in comps:
        y += weight * np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    return y


def gaussian_kde_1d(x, samples, bandwidth):
    """无需外部依赖的简单 1D KDE。"""
    diffs = (x[:, None] - samples[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs**2) / (bandwidth * np.sqrt(2 * np.pi))
    return kernel.mean(axis=1)


def plot_projection_schematic():
    x = np.linspace(-4, 4, 1200)

    # 灰色：训练前/基座策略投影
    p0 = mix_gauss(x, [(0.55, -1.8, 0.9), (0.45, 1.2, 1.1)])

    # 蓝色：SFT 风格（适度锐化）
    psft = mix_gauss(x, [(0.45, -1.8, 1.0), (0.55, 1.5, 0.45)])

    # 蓝色：Off-policy DPO/IPO 风格（更尖、更挤压）
    pdpo = mix_gauss(x, [(0.85, -2.1, 0.22), (0.15, 2.4, 0.40)])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    # 面板 1：SFT
    ax = axes[0]
    ax.plot(x, p0, lw=3, alpha=0.45, color="gray", label="pre")
    ax.plot(x, psft, lw=3, color="tab:blue", label="after")
    ax.set_title("SFT")
    ax.set_yticks([])
    ax.set_xticks([])

    # 标记 y_u^+（红箭头）
    yu_plus = 1.5
    yu_idx = np.argmin(np.abs(x - yu_plus))
    ax.annotate(
        "",
        xy=(yu_plus, psft[yu_idx] * 1.15),
        xytext=(yu_plus, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color="tab:red"),
    )
    ax.text(yu_plus + 0.05, 0.02, r"$y_u^+$")

    # 面板 2：Off-policy DPO/IPO
    ax = axes[1]
    ax.plot(x, p0, lw=3, alpha=0.45, color="gray", label="pre")
    ax.plot(x, pdpo, lw=3, color="tab:blue", label="after")
    ax.set_title("Off-policy DPO / IPO")
    ax.set_yticks([])
    ax.set_xticks([])

    # 标记 y*（虚线）
    y_star = -2.1
    ax.axvline(y_star, ls="--", lw=2, color="black")
    ax.text(y_star - 0.05, 0.02, r"$y^*$")

    # squeezing 区域注释
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
    # 离散候选集：同一组 response 上的对比
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
    ax.set_title("离散候选集策略（同一组 response）")
    ax.set_xlabel("候选 response 排名（按质量排序）")
    ax.set_ylabel("概率质量")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(["pre", "after"], frameon=False)
    fig.tight_layout()


def plot_semantic_projection(data_dir):
    # 语义投影：将 response 投影到 1D 轴并做 KDE
    df = pd.read_csv(data_dir / "semantic_projection.csv")
    pre = df["z_pre"].to_numpy()
    post = df["z_post"].to_numpy()

    x = np.linspace(-4, 4, 800)
    kde_pre = gaussian_kde_1d(x, pre, bandwidth=0.35)
    kde_post = gaussian_kde_1d(x, post, bandwidth=0.3)

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.plot(x, kde_pre, color="gray", lw=3, alpha=0.5, label="pre")
    ax.plot(x, kde_post, color="tab:blue", lw=3, label="after")
    ax.set_title("语义投影 KDE（一维）")
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
