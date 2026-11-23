"""Data and Workshop #2 classifier shared by every Workshop #5 script.

The GANs work in the space of Workshop #2's RoBERTa CLS vectors (768-d), extracted once
in Workshop #4 (`src/extract_embeddings.py` there). The vectors are indexed like the silver
layer, so Workshop #1's frozen split indices select each split directly.

Two spaces are used:
* standardised (StandardScaler fitted on train): what the GANs learn and generate, and
  where the Fréchet distance is measured — every dimension weighs the same;
* raw: what Workshop #2's classifier head reads. Generated vectors are mapped back with
  the scaler's inverse transform before the head sees them.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from config import (EMB_PATH, LABELS, LABELS_PATH, SCALER_PATH, SILVER_PATH, SPLITS_PATH,
                    TEXT_COL, W2_CHECKPOINT, W2_HEAD_PATH)

SPLITS = ("train", "validation", "test")


def load_splits() -> dict[str, dict[str, np.ndarray]]:
    """{split: {"X": raw CLS vectors (float32), "y": label ids, "index": silver row ids}}."""
    for path in (EMB_PATH, LABELS_PATH, SPLITS_PATH):
        if not path.exists():
            msg = f"{path} is missing: see README, section Data."
            raise FileNotFoundError(msg)
    embeddings = np.load(EMB_PATH).astype(np.float32)
    labels = np.load(LABELS_PATH).astype(np.int64)
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    return {name: {"X": embeddings[splits[name]], "y": labels[splits[name]],
                   "index": np.asarray(splits[name])} for name in SPLITS}


def load_scaler(train_X: np.ndarray) -> StandardScaler:
    """StandardScaler fitted on the training vectors only (refitted and saved if absent)."""
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    scaler = StandardScaler().fit(train_X)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_data() -> tuple[dict, StandardScaler]:
    """Splits with an extra "Z" entry (standardised vectors) and the fitted scaler."""
    data = load_splits()
    scaler = load_scaler(data["train"]["X"])
    for split in data.values():
        split["Z"] = scaler.transform(split["X"]).astype(np.float32)
    return data, scaler


def to_raw(scaler: StandardScaler, Z: np.ndarray) -> np.ndarray:
    """Standardised (generated) vectors → the raw space Workshop #2's head reads."""
    return scaler.inverse_transform(Z).astype(np.float32)


def load_texts() -> pd.Series:
    """Review text of every silver row (for nearest-neighbour inspection of generated vectors)."""
    return pd.read_parquet(SILVER_PATH, columns=[TEXT_COL])[TEXT_COL]


# ---------------------------------------------------------------------- Workshop #2 head

def w2_head() -> nn.Module:
    """Workshop #2's classifier on top of the CLS vector: dropout 0.1 + Linear(768, 3)."""
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(768, len(LABELS)))


def load_w2_head() -> nn.Module:
    """Workshop #2's trained head, in eval mode.

    The head weights are cut out of the 499 MB RoBERTa checkpoint once and cached in
    checkpoints/w2_head.pt (12 KB), so the other scripts do not reload the full model.
    """
    if not W2_HEAD_PATH.exists():
        state = torch.load(W2_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
        torch.save({"weight": state["classifier.weight"].clone(),
                    "bias": state["classifier.bias"].clone()}, W2_HEAD_PATH)
    head = w2_head()
    head[1].load_state_dict(torch.load(W2_HEAD_PATH, map_location="cpu"))
    return head.eval()


@torch.no_grad()
def predict_proba(model: nn.Module, X_raw: np.ndarray) -> np.ndarray:
    """Class probabilities of a head over raw CLS vectors."""
    return torch.softmax(model.eval()(torch.from_numpy(X_raw)), dim=-1).numpy()
