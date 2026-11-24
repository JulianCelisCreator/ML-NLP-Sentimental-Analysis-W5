"""Sample generation and visualisation (Workshop #5, artefacts 1–3).

Generated vectors are 768-d embeddings, not images, so the brief's sample grids take two forms:

* tile grids — 8 × 8 grids where each tile is one vector reshaped to 24 × 32 and drawn as a
  heatmap, real and generated side by side. Texture and contrast reveal off-distribution
  vectors; identical tiles reveal mode collapse at a glance;
* review grids — 4 × 4 grids of the real training review closest (cosine) to each generated
  vector, with Workshop #2's predicted label: what the vector "reads like".

Also drawn: training curves of both models, a PCA map of real vs. generated vectors, and
the fixed-noise snapshots saved every 10 epochs projected on that map (training evolution).

    python src/generate_samples.py
"""

from __future__ import annotations

import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from config import (CHECKPOINT_DIR, CLASS_COLORS, FIGURES_DIR, LABELS, MODEL_COLORS, RANDOM_STATE,
                    RUNS_DIR, SAMPLES_DIR)
from data import load_data, load_texts, load_w2_head, predict_proba, to_raw
from evaluate_gan import generate

MODELS = ("vanilla", "wgan", "vanilla_cond", "wgan_cond")
TITLES = {"real": "Real (train)", "vanilla": "Vanilla GAN", "wgan": "WGAN-GP",
          "vanilla_cond": "cGAN (vanilla)", "wgan_cond": "cGAN (WGAN-GP)"}


def available(models=MODELS) -> list[str]:
    return [m for m in models if (CHECKPOINT_DIR / f"gan_{m}_best.pt").exists()]


def history(name: str) -> dict | None:
    path = RUNS_DIR / f"{name}_history.json"
    return json.loads(path.read_text()) if path.exists() else None


def _style(ax, title, xlabel="epoch", ylabel=None):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def training_curves() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    van, wg = history("vanilla"), history("wgan")
    if van:
        h = van["history"]
        e = [r["epoch"] for r in h]
        axes[0, 0].plot(e, [r["d_loss"] for r in h], color="#2a78d6", label="D loss")
        axes[0, 0].plot(e, [r["g_loss"] for r in h], color="#eb6834", label="G loss")
        axes[0, 0].axhline(2 * np.log(2), color="#52514e", ls="--", lw=1)
        axes[0, 0].annotate("D loss at D = 0.5 (2 log 2)", (e[-1], 2 * np.log(2)), xytext=(0, 4),
                            textcoords="offset points", ha="right", fontsize=8, color="#52514e")
        axes[0, 0].legend(frameon=False)
        _style(axes[0, 0], "Vanilla GAN: losses (BCE)")
        axes[0, 1].plot(e, [r["d_real"] for r in h], color="#1baf7a", label="mean D(x), real")
        axes[0, 1].plot(e, [r["d_fake"] for r in h], color="#7c5cc4", label="mean D(G(z)), fake")
        axes[0, 1].plot(e, [r["acc_real"] for r in h], color="#1baf7a", ls=":", label="D accuracy, real")
        axes[0, 1].plot(e, [r["acc_fake"] for r in h], color="#7c5cc4", ls=":", label="D accuracy, fake")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend(frameon=False, fontsize=8)
        _style(axes[0, 1], "Vanilla GAN: discriminator outputs")
    if wg:
        h = wg["history"]
        e = [r["epoch"] for r in h]
        axes[1, 0].plot(e, [r["wasserstein"] for r in h], color="#eb6834", label="W estimate E[C(x)] − E[C(x̃)]")
        axes[1, 0].plot(e, [r["d_loss"] for r in h], color="#2a78d6", label="critic loss")
        axes[1, 0].legend(frameon=False)
        _style(axes[1, 0], "WGAN-GP: critic")
        axes[1, 1].plot(e, [r["gp"] for r in h], color="#7c5cc4")
        _style(axes[1, 1], "WGAN-GP: gradient penalty E[(‖∇C(x̂)‖ − 1)²]")
    for name in available():
        run = history(name)
        if not run:
            continue
        evals = [r for r in run["history"] if "val_fd" in r]
        color = MODEL_COLORS[name]
        axes[0, 2].plot([r["epoch"] for r in evals], [r["val_f1"] for r in evals], marker="o", ms=3,
                        color=color, label=TITLES[name])
        axes[1, 2].plot([r["epoch"] for r in run["history"]], [r["g_grad"] for r in run["history"]],
                        color=color, label=TITLES[name])
    axes[0, 2].legend(frameon=False, fontsize=8)
    _style(axes[0, 2], "Validation generative F1 (precision–recall)")
    axes[1, 2].set_yscale("log")
    _style(axes[1, 2], "Norm of the gradient reaching G")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", dpi=180)
    plt.close(fig)

    # Validation FD / precision / recall over epochs, every model.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in available():
        run = history(name)
        evals = [r for r in run["history"] if "val_fd" in r]
        for ax, key in zip(axes, ("val_fd", "val_precision", "val_recall")):
            ax.plot([r["epoch"] for r in evals], [r[key] for r in evals], marker="o", ms=3,
                    color=MODEL_COLORS[name], label=TITLES[name])
    for ax, title in zip(axes, ("Fréchet distance (validation) ↓", "Precision ↑", "Recall ↑")):
        _style(ax, title)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_metrics.png", dpi=180)
    plt.close(fig)


def pca_maps(data, scaler, head) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    train = data["train"]["Z"]
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(train)
    real_index = rng.choice(len(train), 3000, replace=False)
    background = pca.transform(train[real_index])
    sources = {"real": train[real_index]}
    for name in available():
        sources[name], _ = generate(name, 3000, data["train"]["y"])
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2 * len(sources), 4.2), sharex=True, sharey=True)
    for ax, (name, Z) in zip(axes, sources.items()):
        points = pca.transform(Z)
        predicted = predict_proba(head, to_raw(scaler, Z)).argmax(1)
        ax.scatter(*background.T, s=3, color="#d9d9d6", alpha=0.6, rasterized=True)
        for index, label in enumerate(LABELS):
            mask = predicted == index
            ax.scatter(*points[mask].T, s=4, alpha=0.6, color=CLASS_COLORS[label], rasterized=True,
                       label=f"{label} ({mask.mean():.0%})")
        ax.set_title(TITLES[name], fontsize=11)
        ax.legend(frameon=False, fontsize=7, markerscale=3, loc="lower right")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("PCA of the standardised CLS vectors (grey: real training vectors; colour: class "
                 "predicted by Workshop #2's head)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_real_vs_generated.png", dpi=180)
    plt.close(fig)

    # Training evolution: fixed-noise snapshots projected on the same map.
    names = [n for n in ("vanilla", "wgan") if (SAMPLES_DIR / n).exists()]
    epochs = (1, 10, 30, 70, 150)
    fig, axes = plt.subplots(len(names), len(epochs), figsize=(3.2 * len(epochs), 3.2 * len(names)),
                             sharex=True, sharey=True, squeeze=False)
    for row, name in enumerate(names):
        for col, epoch in enumerate(epochs):
            ax = axes[row, col]
            ax.scatter(*background.T, s=2, color="#d9d9d6", alpha=0.6, rasterized=True)
            path = SAMPLES_DIR / name / f"epoch_{epoch:03d}.npy"
            if path.exists():
                ax.scatter(*pca.transform(np.load(path).astype(np.float32)).T, s=10,
                           color=MODEL_COLORS[name])
            ax.set_title(f"{TITLES[name]} · epoch {epoch}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Same 64 noise vectors through training (grey: real vectors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_evolution.png", dpi=180)
    plt.close(fig)


def tile_grids(data) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    train = data["train"]["Z"]
    sources = {"real": train[rng.choice(len(train), 64, replace=False)]}
    for name in available(("vanilla", "wgan")):
        sources[name], _ = generate(name, 64, data["train"]["y"], seed=7)
    fig, axes = plt.subplots(1, len(sources), figsize=(5.4 * len(sources), 4.9), layout="constrained")
    for ax, (name, Z) in zip(axes, sources.items()):
        tiles = np.clip(Z, -3, 3).reshape(8, 8, 24, 32)
        canvas = np.full((8 * 25 - 1, 8 * 33 - 1), np.nan)
        for i in range(8):
            for j in range(8):
                canvas[i * 25:i * 25 + 24, j * 33:j * 33 + 32] = tiles[i, j]
        image = ax.imshow(canvas, cmap="RdBu_r", vmin=-3, vmax=3)
        ax.set_title(f"{TITLES[name]} — 64 vectors", fontsize=12)
        ax.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.8, label="standardised value (clipped to ±3)")
    fig.suptitle("8 × 8 sample grids: each tile is one 768-d vector drawn as a 24 × 32 heatmap",
                 fontsize=12)
    fig.savefig(FIGURES_DIR / "sample_grids.png", dpi=160)
    plt.close(fig)


def review_grids(data, scaler, head) -> None:
    """Nearest real review to 16 vectors per source; also written to samples_nearest_reviews.md."""
    texts = load_texts()
    train = torch.from_numpy(data["train"]["Z"])
    train_unit = torch.nn.functional.normalize(train, dim=1)
    rng = np.random.default_rng(RANDOM_STATE)
    sources = {"real": data["validation"]["Z"][rng.choice(len(data["validation"]["Z"]), 16, replace=False)]}
    for name in available():
        sources[name], _ = generate(name, 16, data["train"]["y"], seed=11)
    lines = ["# Nearest real training review to generated vectors", "",
             "Cosine similarity in the standardised CLS space; label = Workshop #2's prediction "
             "for the generated vector. For \"real\", the 16 vectors are validation reviews "
             "(the similarity a genuine unseen review gets).", ""]
    fig, axes = plt.subplots(len(sources), 1, figsize=(18, 5.2 * len(sources)))
    for ax, (name, Z) in zip(np.atleast_1d(axes), sources.items()):
        query = torch.nn.functional.normalize(torch.from_numpy(Z), dim=1)
        similarity, nearest = (query @ train_unit.T).max(1)
        predicted = predict_proba(head, to_raw(scaler, Z)).argmax(1)
        lines += [f"## {'Real (validation)' if name == 'real' else TITLES[name]}", "", "| # | Predicted | Cosine | Nearest training review (truncated) |",
                  "| --- | --- | --- | --- |"]
        ax.axis("off")
        title = "Real (validation)" if name == "real" else TITLES[name]
        ax.set_title(f"{title} — nearest real review to 16 vectors", fontsize=13, loc="left")
        for k in range(16):
            text = str(texts.iloc[int(data["train"]["index"][int(nearest[k])])]).replace("\n", " ")
            label = LABELS[predicted[k]]
            lines.append(f"| {k + 1} | {label} | {similarity[k]:.2f} | {text[:220].replace('|', '/')}… |")
            row, col = divmod(k, 4)
            snippet = "\n".join(textwrap.wrap(text[:150] + "…", 42)[:4])
            ax.text(col / 4 + 0.005, 1 - row / 4 - 0.02, f"{label} · cos {similarity[k]:.2f}",
                    transform=ax.transAxes, parse_math=False, fontsize=9, fontweight="bold", color=CLASS_COLORS[label],
                    va="top")
            ax.text(col / 4 + 0.005, 1 - row / 4 - 0.065, snippet, transform=ax.transAxes, parse_math=False,
                    fontsize=8, va="top", color="#2b2b2b")
        lines.append("")
    fig.savefig(FIGURES_DIR / "review_grids.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    (RUNS_DIR / "samples_nearest_reviews.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    torch.set_num_threads(6)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data, scaler = load_data()
    head = load_w2_head()
    training_curves()
    pca_maps(data, scaler, head)
    tile_grids(data)
    review_grids(data, scaler, head)
    print(f"Figures written to {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
