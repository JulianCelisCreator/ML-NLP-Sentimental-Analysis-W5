"""Generation quality of the trained GANs (Workshop #5, artefact 3).

Every model generates 10,000 standardised vectors, compared with the held-out real test
split (the GANs never see it; validation is used for checkpoint selection). Metrics:

* Fréchet distance (FD) between Gaussians fitted to real and generated vectors: the FID
  formula with RoBERTa CLS vectors in place of Inception features. Lower is better.
* Classifier score: the Inception Score formula, exp(E[KL(p(y|x) ‖ p(y))]), with Workshop
  #2's sentiment head as the classifier. Range 1–3 (three classes); compare with real data.
* Precision and recall (Kynkäänniemi et al., 2019, k = 5): the share of generated vectors
  inside the real data's k-NN manifold (fidelity) and of real vectors inside the generated
  one (coverage — what mode collapse destroys).
* Diversity: mean pairwise distance, generated vs. real; effective dimension (participation
  ratio of the covariance spectrum), which exposes samples spread over too few directions.
* Memorisation: distance from each vector to its nearest training vector, generated vs.
  real test reviews (a ratio well below 1 would mean G copies its training set).
* Class mix under Workshop #2's head, and for the cGANs the label fidelity: macro-F1 of the
  head against the label each vector was generated for.

Two reference rows frame the numbers: "real" (the 6,395 held-out validation vectors: the
score of genuine unseen reviews, the best any generator can do) and "gaussian" (a multivariate normal fitted to the training vectors —
it matches the two moments FD looks at, so it shows what FD alone cannot see).

    python src/evaluate_gan.py
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy import linalg
from sklearn.metrics import f1_score

from config import CHECKPOINT_DIR, LABELS, RANDOM_STATE, RUNS_DIR
from data import load_data, load_w2_head, predict_proba, to_raw

MODELS = ("vanilla", "wgan", "vanilla_cond", "wgan_cond")
# Failure-mode ablations of the vanilla GAN (src/train_gan.py --tag …), evaluated the same way.
ABLATIONS = ("vanilla_mbstd", "vanilla_noise", "vanilla_strongd", "vanilla_unstable")
N_EVAL = 10_000


# ---------------------------------------------------------------------- metrics

def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """‖µa − µb‖² + Tr(Σa + Σb − 2(Σa Σb)^½).

    Tr((Σa Σb)^½) is computed as Σ √λ of the symmetric PSD matrix Σa^½ Σb Σa^½ (same
    eigenvalues as Σa Σb), which avoids the complex round-off of a general matrix sqrt.
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    values, vectors = linalg.eigh(cov_a)
    sqrt_a = (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T
    middle = linalg.eigvalsh(sqrt_a @ cov_b @ sqrt_a)
    trace_covmean = np.sqrt(np.clip(middle, 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(cov_a) + np.trace(cov_b) - 2 * trace_covmean)


def classifier_score(probabilities: np.ndarray) -> float:
    """Inception-Score formula with any classifier's probabilities p(y | x)."""
    marginal = probabilities.mean(0, keepdims=True)
    kl = (probabilities * (np.log(probabilities + 1e-12) - np.log(marginal + 1e-12))).sum(1)
    return float(np.exp(kl.mean()))


def _kth_radius(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each row to its k-th nearest other row."""
    radii = []
    for chunk in x.split(1024):
        d = torch.cdist(chunk, x)
        radii.append(d.kthvalue(k + 1, dim=1).values)      # +1: the row itself is at distance 0
    return torch.cat(radii)


def _inside(points: torch.Tensor, centres: torch.Tensor, radii: torch.Tensor) -> float:
    """Share of `points` inside at least one ball (centre, radius)."""
    hits = [(torch.cdist(chunk, centres) <= radii).any(1) for chunk in points.split(1024)]
    return float(torch.cat(hits).float().mean())


def precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 5, n: int = 5000,
                     seed: int = RANDOM_STATE) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = torch.from_numpy(real[rng.choice(len(real), min(n, len(real)), replace=False)])
    f = torch.from_numpy(fake[rng.choice(len(fake), min(n, len(fake)), replace=False)])
    return _inside(f, r, _kth_radius(r, k)), _inside(r, f, _kth_radius(f, k))


def mean_pairwise_distance(x: np.ndarray, n: int = 2000, seed: int = RANDOM_STATE) -> float:
    rng = np.random.default_rng(seed)
    t = torch.from_numpy(x[rng.choice(len(x), min(n, len(x)), replace=False)])
    d = torch.cdist(t, t)
    return float(d.sum() / (len(t) * (len(t) - 1)))


def effective_dimension(x: np.ndarray) -> float:
    """Participation ratio (Σλ)² / Σλ² of the covariance spectrum: how many directions the
    vectors really spread over. A generator can match the overall spread (diversity) yet put
    its samples on a thin, low-dimensional sheet — a partial collapse that recall exposes."""
    eigenvalues = np.clip(linalg.eigvalsh(np.cov(x.astype(np.float64), rowvar=False)), 0, None)
    return float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())


def nearest_train_distance(x: np.ndarray, train: np.ndarray, n: int = 2000,
                           seed: int = RANDOM_STATE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = torch.from_numpy(x[rng.choice(len(x), min(n, len(x)), replace=False)])
    reference = torch.from_numpy(train)
    return torch.cat([torch.cdist(chunk, reference).min(1).values for chunk in t.split(512)]).numpy()


# ---------------------------------------------------------------------- generation

def class_mix(y_train: np.ndarray) -> np.ndarray:
    """Share of each class in the real training set."""
    return np.bincount(y_train, minlength=len(LABELS)) / len(y_train)


def proportional_labels(count: int, y_train: np.ndarray, seed: int = RANDOM_STATE) -> torch.Tensor:
    """Labels for a conditional generator, drawn with the real training class mix."""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.choice(len(LABELS), size=count, p=class_mix(y_train)))


def generate(name: str, count: int, y_train: np.ndarray, seed: int = RANDOM_STATE):
    """(standardised vectors, conditioning labels or None) from checkpoints/gan_<name>_best.pt."""
    from gan import checkpoint_generator, sample

    generator, config = checkpoint_generator(CHECKPOINT_DIR / f"gan_{name}_best.pt")
    labels = proportional_labels(count, y_train, seed) if config.conditional else None
    return sample(generator, count, labels, generator_seed=seed).numpy(), (
        labels.numpy() if labels is not None else None)


def describe(Z: np.ndarray, data: dict, scaler, head, labels=None) -> dict:
    """Every metric for one set of standardised vectors."""
    real_test, real_train = data["test"]["Z"], data["train"]["Z"]
    probabilities = predict_proba(head, to_raw(scaler, Z))
    predictions = probabilities.argmax(1)
    precision, recall = precision_recall(real_test, Z)
    nn_fake = nearest_train_distance(Z, real_train)
    result = {
        "frechet_distance": frechet_distance(real_test, Z),
        "classifier_score": classifier_score(probabilities),
        "precision": precision, "recall": recall,
        "diversity": mean_pairwise_distance(Z),
        "effective_dimension": effective_dimension(Z),
        "nearest_train_median": float(np.median(nn_fake)),
        "class_mix": {label: float((predictions == i).mean()) for i, label in enumerate(LABELS)},
        "mean_confidence": float(probabilities.max(1).mean()),
    }
    if labels is not None:
        result["label_fidelity_macro_f1"] = float(f1_score(labels, predictions, average="macro"))
        result["label_fidelity_accuracy"] = float((labels == predictions).mean())
    return result


def main() -> int:
    torch.set_num_threads(6)
    data, scaler = load_data()
    head = load_w2_head()
    y_train = data["train"]["y"]
    rng = np.random.default_rng(RANDOM_STATE)

    rows = {}
    # Held-out real reviews (validation: never trained on, not the test reference itself).
    rows["real"] = describe(data["validation"]["Z"], data, scaler, head, labels=data["validation"]["y"])

    mean, cov = data["train"]["Z"].mean(0), np.cov(data["train"]["Z"], rowvar=False)
    gaussian = rng.multivariate_normal(mean, cov, size=N_EVAL, method="cholesky").astype(np.float32)
    rows["gaussian"] = describe(gaussian, data, scaler, head)

    for name in MODELS + ABLATIONS:
        if not (CHECKPOINT_DIR / f"gan_{name}_best.pt").exists():
            print(f"  skip {name}: no checkpoint")
            continue
        Z, labels = generate(name, N_EVAL, y_train)
        rows[name] = describe(Z, data, scaler, head, labels)
        print(f"  evaluated {name}")

    real_test_nn = nearest_train_distance(data["test"]["Z"], data["train"]["Z"])
    reference = {"real_test_nearest_train_median": float(np.median(real_test_nn)),
                 "real_test_diversity": mean_pairwise_distance(data["test"]["Z"]),
                 "real_test_effective_dimension": effective_dimension(data["test"]["Z"]),
                 "real_train_class_mix": dict(zip(LABELS, class_mix(y_train).tolist()))}
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / "evaluation.json").write_text(json.dumps({"n_samples": N_EVAL, "reference": reference,
                                                          "models": rows}, indent=2), encoding="utf-8")

    header = ("| Source | FD ↓ | Classifier score | Precision ↑ | Recall ↑ | Diversity | Eff. dim | "
              "NN-train median | % neg / neu / pos | Label F1 |")
    lines = [header, "| --- " * 10 + "|"]
    for name, r in rows.items():
        mix = " / ".join(f"{100 * r['class_mix'][label]:.1f}" for label in LABELS)
        fidelity = f"{r['label_fidelity_macro_f1']:.3f}" if "label_fidelity_macro_f1" in r else "—"
        lines.append(f"| {name} | {r['frechet_distance']:.1f} | {r['classifier_score']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['diversity']:.2f} | "
                     f"{r['effective_dimension']:.1f} | "
                     f"{r['nearest_train_median']:.2f} | {mix} | {fidelity} |")
    lines.append("")
    lines.append(f"Real test reviews: diversity {reference['real_test_diversity']:.2f}, "
                 f"nearest-train median {reference['real_test_nearest_train_median']:.2f}.")
    table = "\n".join(lines)
    (RUNS_DIR / "evaluation.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
