"""Paths, constants and default hyperparameters shared by every Workshop #5 script.

One place for every number the report quotes, so training, evaluation and the
augmentation experiment cannot drift apart.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

# Anchored to the package, not the working directory, so every script resolves the same files.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EMB_PATH = DATA_DIR / "gold" / "embeddings_bert.npy"
LABELS_PATH = DATA_DIR / "gold" / "labels.npy"
SPLITS_PATH = DATA_DIR / "gold" / "splits.json"
SILVER_PATH = DATA_DIR / "silver" / "reviews.parquet"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
W2_CHECKPOINT = CHECKPOINT_DIR / "model_best.pt"     # Workshop #2 RoBERTa (499 MB, not versioned)
W2_HEAD_PATH = CHECKPOINT_DIR / "w2_head.pt"         # its classifier head alone (12 KB)
SCALER_PATH = CHECKPOINT_DIR / "scaler.pkl"

RUNS_DIR = PROJECT_ROOT / "runs" / "workshop5"
SAMPLES_DIR = RUNS_DIR / "samples"
FIGURES_DIR = RUNS_DIR / "figures"

RANDOM_STATE = 42
TEXT_COL = "Review_Text"
# Same order as Workshops #1–#4, so confusion matrices and reports line up.
LABELS = ["negative", "neutral", "positive"]
# Same colour per class as the earlier figures (palette checked for colour blindness).
CLASS_COLORS = dict(zip(LABELS, ["#2a78d6", "#eb6834", "#1baf7a"]))
MODEL_COLORS = {"real": "#52514e", "vanilla": "#2a78d6", "wgan": "#eb6834",
                "vanilla_cond": "#7c5cc4", "wgan_cond": "#1baf7a"}

EMB_DIM = 768
# Workshop #2's reference result on the test split (its head on these same vectors).
W2_TEST_MACRO_F1 = 0.7130


@dataclass
class GANConfig:
    """Every hyperparameter of a GAN run. Saved inside the checkpoint and the run history."""

    model: str = "vanilla"            # "vanilla" (BCE) or "wgan" (WGAN-GP)
    conditional: bool = False         # cGAN: the label is an input of G and D
    z_dim: int = 128
    g_hidden: tuple[int, ...] = (256, 512, 1024)
    d_hidden: tuple[int, ...] = (512, 256)
    label_dim: int = 16               # size of the learned label embedding (cGAN only)
    epochs: int = 150
    batch_size: int = 256
    # Vanilla GAN, as the brief prescribes: Adam 2e-4 for both nets, real labels = 0.9.
    # beta1 = 0.5 is the DCGAN setting.
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    real_label: float = 0.9
    d_dropout: float = 0.3
    # Mode-collapse mitigations, off in the main runs (ablations in the report):
    # instance noise — Gaussian noise of std `instance_noise` added to D's inputs (real and
    # fake), annealed linearly to 0 over training; minibatch standard deviation feature in D.
    instance_noise: float = 0.0
    minibatch_std: bool = False
    # WGAN-GP, as in Gulrajani et al. (2017): 5 critic steps per G step, lambda = 10,
    # Adam 1e-4 with betas (0, 0.9).
    n_critic: int = 5
    gp_lambda: float = 10.0
    wgan_lr: float = 1e-4
    wgan_betas: tuple[float, float] = (0.0, 0.9)
    # Every `eval_every` epochs: Fréchet distance on validation + a saved sample snapshot.
    eval_every: int = 10
    eval_samples: int = 5000
    seed: int = RANDOM_STATE

    @property
    def name(self) -> str:
        return f"{self.model}{'_cond' if self.conditional else ''}"

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
