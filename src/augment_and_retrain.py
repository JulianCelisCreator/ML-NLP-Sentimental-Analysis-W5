"""Data augmentation with the GANs, evaluated on Workshop #2's classifier (artefact 4).

The GANs generate RoBERTa CLS vectors, not text. RoBERTa itself cannot be retrained on them
(it reads tokens); what is retrained is Workshop #2's classification head (dropout 0.1 +
Linear 768 → 3) on top of the frozen fine-tuned encoder — the same experiment as Workshop
#4's VAE augmentation, so the two generative families compare directly.

Conditions, each with 1,000 and 5,000 synthetic vectors added to the 29,842 real ones:
  (a) real            real training vectors only (baseline)
  (b) vanilla         + vanilla GAN vectors
  (c) wgan            + WGAN-GP vectors
  (d) vanilla_cond    + cGAN vectors (vanilla loss)
  (e) wgan_cond       + conditional WGAN-GP vectors
  (f) wgan_cond_minority  + conditional WGAN-GP vectors of the two minority classes only,
                      half negative and half neutral (rebalancing instead of mimicking)

Labels: an unconditional GAN produces unlabeled vectors, so (b) and (c) are labeled by
Workshop #2's frozen head (pseudo-labels — the head trained on real data acts as a teacher).
The conditional GANs (d–f) generate for a requested label, which becomes the vector's label;
(d) and (e) request labels with the real class mix.

Fair comparison: same head, hyperparameters and budget (20 epochs, AdamW 1e-3, batch 256,
class weights from the REAL training labels), epoch chosen on validation macro-F1, three
seeds per cell. Validation and test are 100 % real and never change.

    python src/augment_and_retrain.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from config import (CHECKPOINT_DIR, FIGURES_DIR, LABELS, MODEL_COLORS, PROJECT_ROOT, RUNS_DIR,
                    W2_TEST_MACRO_F1, set_seed)
from data import load_data, load_w2_head, predict_proba, to_raw, w2_head
from gan import checkpoint_generator, sample

COUNTS = (1000, 5000)
SEEDS = (42, 43, 44)
CONDITIONS = ("vanilla", "wgan", "vanilla_cond", "wgan_cond", "wgan_cond_minority")
W4_RESULTS = PROJECT_ROOT.parent / "cuarto_taller" / "runs" / "workshop4" / "augmentation.json"


def scores(model, X, y) -> dict:
    probabilities = predict_proba(model, X)
    predictions = probabilities.argmax(1)
    per_class = f1_score(y, predictions, average=None, labels=range(len(LABELS)))
    return {"macro_f1": float(f1_score(y, predictions, average="macro")),
            "accuracy": float(accuracy_score(y, predictions)),
            "auc_roc": float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro")),
            "f1_per_class": dict(zip(LABELS, map(float, per_class)))}


def train_head(X, y, X_val, y_val, class_weight, seed, epochs=20, lr=1e-3, batch_size=256):
    """Workshop #4's protocol: weighted cross-entropy, best epoch on validation macro-F1."""
    set_seed(seed)
    model = w2_head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    X_t, y_t = torch.from_numpy(X), torch.from_numpy(y)
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for index in torch.randperm(len(X_t)).split(batch_size):
            loss = criterion(model(X_t[index]), y_t[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        f1 = scores(model, X_val, y_val)["macro_f1"]
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def synthetic_set(condition: str, count: int, seed: int, y_train, scaler, teacher):
    """(raw vectors, labels) for one condition."""
    name = condition.removesuffix("_minority")
    generator, config = checkpoint_generator(CHECKPOINT_DIR / f"gan_{name}_best.pt")
    rng = np.random.default_rng(seed)
    labels = None
    if config.conditional:
        if condition.endswith("_minority"):
            labels = rng.permutation(np.arange(count) % 2)            # negative / neutral, half each
        else:
            mix = np.bincount(y_train, minlength=len(LABELS)) / len(y_train)
            labels = rng.choice(len(LABELS), size=count, p=mix)
    Z = sample(generator, count, torch.from_numpy(labels) if labels is not None else None,
               generator_seed=seed).numpy()
    X = to_raw(scaler, Z)
    if labels is None:
        labels = predict_proba(teacher, X).argmax(1)                   # pseudo-labels
    return X, labels.astype(np.int64)


def main() -> int:
    torch.set_num_threads(6)
    data, scaler = load_data()
    X_train, y_train = data["train"]["X"], data["train"]["y"]
    X_val, y_val = data["validation"]["X"], data["validation"]["y"]
    X_test, y_test = data["test"]["X"], data["test"]["y"]
    teacher = load_w2_head()
    sanity = scores(teacher, X_test, y_test)
    print(f"Workshop #2 head on the test vectors: macro-F1 {sanity['macro_f1']:.4f} "
          f"(expected {W2_TEST_MACRO_F1:.4f})")
    class_weight = torch.tensor(compute_class_weight(
        "balanced", classes=np.arange(len(LABELS)), y=y_train), dtype=torch.float32)

    runs = []
    cells = [("real", 0)] + [(c, n) for c in CONDITIONS for n in COUNTS]
    for condition, count in cells:
        for seed in SEEDS:
            X, y = X_train, y_train
            mix = None
            if count:
                X_syn, y_syn = synthetic_set(condition, count, seed, y_train, scaler, teacher)
                X, y = np.concatenate([X_train, X_syn]), np.concatenate([y_train, y_syn])
                mix = dict(zip(LABELS, (np.bincount(y_syn, minlength=3) / len(y_syn)).tolist()))
            model = train_head(X, y, X_val, y_val, class_weight, seed)
            run = {"condition": condition, "synthetic": count, "seed": seed, "train_rows": int(len(X)),
                   "synthetic_label_mix": mix, "validation": scores(model, X_val, y_val),
                   "test": scores(model, X_test, y_test)}
            runs.append(run)
            print(f"  {condition:<19} +{count:>5}  seed {seed}  val {run['validation']['macro_f1']:.4f}"
                  f"  test {run['test']['macro_f1']:.4f}", flush=True)

    summary = {}
    for condition, count in cells:
        cell = [r for r in runs if r["condition"] == condition and r["synthetic"] == count]
        f1 = [r["test"]["macro_f1"] for r in cell]
        summary[f"{condition}+{count}"] = {
            "condition": condition, "synthetic": count,
            "test_macro_f1_mean": float(np.mean(f1)), "test_macro_f1_std": float(np.std(f1)),
            "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in cell])),
            "test_auc_mean": float(np.mean([r["test"]["auc_roc"] for r in cell])),
            "test_f1_per_class_mean": {label: float(np.mean([r["test"]["f1_per_class"][label] for r in cell]))
                                       for label in LABELS}}
    w4 = json.loads(W4_RESULTS.read_text()) ["summary"] if W4_RESULTS.exists() else None
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "augmentation.json").write_text(json.dumps({
        "w2_head_on_test": sanity, "seeds": SEEDS, "summary": summary,
        "workshop4_vae_summary": w4, "runs": runs}, indent=2), encoding="utf-8")

    base = summary["real+0"]["test_macro_f1_mean"]
    lines = ["| Condition | Synthetic | Test macro-F1 (mean ± std) | Δ vs. real | Accuracy | AUC-ROC | "
             "F1 negative | F1 neutral | F1 positive |", "| --- " * 9 + "|"]
    for value in summary.values():
        pc = value["test_f1_per_class_mean"]
        lines.append(f"| {value['condition']} | {value['synthetic']:,} | {value['test_macro_f1_mean']:.4f} ± "
                     f"{value['test_macro_f1_std']:.4f} | {value['test_macro_f1_mean'] - base:+.4f} | "
                     f"{value['test_accuracy_mean']:.4f} | {value['test_auc_mean']:.4f} | "
                     f"{pc['negative']:.3f} | {pc['neutral']:.3f} | {pc['positive']:.3f} |")
    table = "\n".join(lines)
    (RUNS_DIR / "augmentation.md").write_text(table + "\n", encoding="utf-8")
    print(table)

    # Figure: Δ test macro-F1 per condition and size, with the seed spread.
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.38
    for offset, count in zip((-width / 2, width / 2), COUNTS):
        means = [summary[f"{c}+{count}"]["test_macro_f1_mean"] - base for c in CONDITIONS]
        stds = [summary[f"{c}+{count}"]["test_macro_f1_std"] for c in CONDITIONS]
        colors = [MODEL_COLORS.get(c.removesuffix("_minority"), "#52514e") for c in CONDITIONS]
        bars = ax.bar(np.arange(len(CONDITIONS)) + offset, means, width, yerr=stds, capsize=3,
                      color=colors, alpha=0.55 if count == COUNTS[0] else 1.0,
                      edgecolor="#2b2b2b", linewidth=0.6)
        for bar, value in zip(bars, means):
            ax.annotate(f"{value:+.4f}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3 if value >= 0 else -11), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.axhline(0, color="#52514e", linewidth=1)
    ax.set_xticks(np.arange(len(CONDITIONS)), [c.replace("_", "\n") for c in CONDITIONS])
    ax.set_ylabel("Δ test macro-F1 vs. real only")
    ax.set_title(f"GAN augmentation of Workshop #2's head (baseline macro-F1 {base:.4f}); "
                 f"light bars +{COUNTS[0]:,}, dark bars +{COUNTS[1]:,}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "augmentation.png", dpi=200, bbox_inches="tight")
    print(f"Saved {RUNS_DIR / 'augmentation.json'}, augmentation.md and figures/augmentation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
