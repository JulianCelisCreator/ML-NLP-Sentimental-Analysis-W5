"""Generator, Discriminator and Critic for Workshop #5.

The data are 768-d RoBERTa CLS vectors (standardised), i.e. tabular-like continuous data,
so the brief's guidance for tabular data applies: fully-connected layers in every network.
The DCGAN guidelines carry over to the dense case: BatchNorm + ReLU in the generator,
LeakyReLU(0.2) in the discriminator, no BatchNorm in the discriminator's input layer.

* Generator     z (128) → 256 → 512 → 1024 → 768. Linear output: the standardised vectors
                are unbounded (mean 0, std 1 per dimension), so a Tanh would clip their tails.
* Discriminator 768 → 512 → 256 → 1 logit. The Sigmoid is fused into the loss
                (BCEWithLogitsLoss = Sigmoid + BCE, numerically stable); `probability()`
                applies it explicitly.
* Critic        same layers, no Sigmoid: a real-valued score (WGAN). LayerNorm instead of
                dropout/BatchNorm, because the gradient penalty is computed per sample and
                BatchNorm would mix the samples of a batch.

Conditional variants (cGAN, the optional part of the brief): a learned label embedding is
concatenated to z in the generator and to the input vector in the discriminator/critic.

    python src/gan.py      # shape and parameter check
"""

from __future__ import annotations

import torch
from torch import nn

from config import EMB_DIM, GANConfig, LABELS


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, hidden: tuple[int, ...] = (256, 512, 1024),
                 out_dim: int = EMB_DIM, num_classes: int = 0, label_dim: int = 16):
        super().__init__()
        self.z_dim, self.num_classes = z_dim, num_classes
        self.label_embedding = nn.Embedding(num_classes, label_dim) if num_classes else None
        layers, width = [], z_dim + (label_dim if num_classes else 0)
        for size in hidden:
            layers += [nn.Linear(width, size), nn.BatchNorm1d(size), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.label_embedding is not None:
            z = torch.cat([z, self.label_embedding(labels)], dim=1)
        return self.net(z)


class Discriminator(nn.Module):
    """Vanilla GAN discriminator (critic=False) or WGAN critic (critic=True)."""

    def __init__(self, in_dim: int = EMB_DIM, hidden: tuple[int, ...] = (512, 256),
                 num_classes: int = 0, label_dim: int = 16, critic: bool = False,
                 dropout: float = 0.3):
        super().__init__()
        self.critic = critic
        self.label_embedding = nn.Embedding(num_classes, label_dim) if num_classes else None
        layers, width = [], in_dim + (label_dim if num_classes else 0)
        for size in hidden:
            layers.append(nn.Linear(width, size))
            if critic:
                layers.append(nn.LayerNorm(size))
            layers.append(nn.LeakyReLU(0.2))
            if not critic:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """Logit (discriminator) or score (critic), shape (batch,)."""
        if self.label_embedding is not None:
            x = torch.cat([x, self.label_embedding(labels)], dim=1)
        return self.net(x).squeeze(1)

    def probability(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        """D(x) ∈ (0, 1): the Sigmoid output of the vanilla discriminator."""
        return torch.sigmoid(self(x, labels))


def build(config: GANConfig) -> tuple[Generator, Discriminator]:
    """The generator and discriminator/critic a config describes. Same G for both models."""
    num_classes = len(LABELS) if config.conditional else 0
    generator = Generator(config.z_dim, config.g_hidden, num_classes=num_classes,
                          label_dim=config.label_dim)
    discriminator = Discriminator(hidden=config.d_hidden, num_classes=num_classes,
                                  label_dim=config.label_dim, critic=config.model == "wgan",
                                  dropout=config.d_dropout)
    return generator, discriminator


def gradient_penalty(critic: Discriminator, real: torch.Tensor, fake: torch.Tensor,
                     labels: torch.Tensor | None = None) -> torch.Tensor:
    """E[(‖∇ C(x̂)‖₂ − 1)²] at x̂ = ε·x + (1 − ε)·x̃, ε ~ U[0, 1] per sample (λ applied by the caller)."""
    epsilon = torch.rand(real.size(0), 1)
    mixed = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    scores = critic(mixed, labels)
    # create_graph=True: the penalty is itself differentiated when the critic is updated.
    gradients, = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def sample(generator: Generator, count: int, labels: torch.Tensor | None = None,
           generator_seed: int | None = None, batch_size: int = 2048) -> torch.Tensor:
    """`count` generated (standardised) vectors; `labels` is required for a conditional G."""
    rng = torch.Generator().manual_seed(generator_seed) if generator_seed is not None else None
    generator.eval()
    parts = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        z = torch.randn(size, generator.z_dim, generator=rng)
        y = labels[start:start + size] if labels is not None else None
        parts.append(generator(z, y))
    return torch.cat(parts)


def checkpoint_generator(path) -> tuple[Generator, GANConfig]:
    """Rebuild a generator saved by src/train_gan.py."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(checkpoint["config"])
    for key in ("g_hidden", "d_hidden", "betas", "wgan_betas"):
        values[key] = tuple(values[key])
    config = GANConfig(**values)
    generator, _ = build(config)
    generator.load_state_dict(checkpoint["generator"])
    return generator.eval(), config


if __name__ == "__main__":
    for model in ("vanilla", "wgan"):
        for conditional in (False, True):
            cfg = GANConfig(model=model, conditional=conditional)
            g, d = build(cfg)
            y = torch.randint(0, len(LABELS), (8,)) if conditional else None
            x = g(torch.randn(8, cfg.z_dim), y)
            out = d(x, y)
            print(f"{cfg.name:<13} G {sum(p.numel() for p in g.parameters()):>9,} params → {tuple(x.shape)}"
                  f" | D {sum(p.numel() for p in d.parameters()):>9,} params → {tuple(out.shape)}")
