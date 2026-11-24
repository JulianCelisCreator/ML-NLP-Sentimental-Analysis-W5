"""Training loop for the vanilla GAN and WGAN-GP (Workshop #5, artefacts 1 and 2).

Both models learn the distribution of Workshop #2's standardised RoBERTa CLS vectors of the
training split. Per epoch the run logs the losses, the discriminator's accuracy on real and
fake vectors (vanilla) or the Wasserstein estimate and gradient penalty (WGAN-GP), and the
norm of the gradient reaching the generator (the vanishing-gradient symptom). Every 10
epochs it measures the Fréchet distance, precision and recall of 5,000 generated vectors
against the real validation split and saves 64 vectors from a fixed noise batch (the
"sample grid" at that epoch). GAN losses do not track sample quality, so the last epoch is
not necessarily the best: the checkpoint kept is the one with the highest validation
"generative F1", the harmonic mean of precision and recall. Not FD alone: in this space a
plain Gaussian fitted to the training vectors already reaches the real-vs-real FD (7.0 vs.
6.8) while only 9 % of its samples are realistic (precision 0.09).

Vanilla GAN: BCE, Adam 2e-4 (β = 0.5, 0.999) for G and D, one-sided label smoothing (real
labels 0.9), non-saturating generator loss −log D(G(z)), one D step per G step.
WGAN-GP: critic without Sigmoid, loss E[C(x̃)] − E[C(x)] + λ·GP with λ = 10, five critic
steps per generator step, Adam 1e-4 (β = 0, 0.9), generator loss −E[C(G(z))].

    python src/train_gan.py --model vanilla
    python src/train_gan.py --model wgan
    python src/train_gan.py --model wgan --conditional          # cGAN (optional part)
    python src/train_gan.py --model vanilla --real-label 1.0 --d-lr 8e-4 --tag unstable
    python src/train_gan.py --model vanilla --minibatch-std --tag mbstd   # mode-collapse fixes
    python src/train_gan.py --model vanilla --d-dropout 0 --d-lr 4e-4 --tag strongd
"""

from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
from torch import nn

from config import CHECKPOINT_DIR, GANConfig, RUNS_DIR, SAMPLES_DIR, set_seed
from data import load_data
from evaluate_gan import frechet_distance, precision_recall
from gan import build, gradient_penalty, sample

SNAPSHOT_SIZE = 64          # an 8 × 8 grid


def grad_norm(module: nn.Module) -> float:
    """L2 norm of the gradient currently stored in a module's parameters."""
    norms = [p.grad.norm(2) for p in module.parameters() if p.grad is not None]
    return float(torch.norm(torch.stack(norms), 2)) if norms else 0.0


def evaluate(generator, config, Z_val, y_val, y_train) -> dict:
    """FD, precision and recall of `eval_samples` generated vectors vs. the validation split."""
    labels = None
    if config.conditional:
        rng = np.random.default_rng(config.seed)
        mix = np.bincount(y_train, minlength=3) / len(y_train)
        labels = torch.from_numpy(rng.choice(3, size=config.eval_samples, p=mix))
    fake = sample(generator, config.eval_samples, labels, generator_seed=config.seed).numpy()
    precision, recall = precision_recall(Z_val, fake, n=config.eval_samples)
    generator.train()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"val_fd": frechet_distance(Z_val, fake), "val_precision": precision,
            "val_recall": recall, "val_f1": f1}


def train(config: GANConfig, tag: str = "", d_lr: float | None = None) -> dict:
    set_seed(config.seed)
    data, _ = load_data()
    X = torch.from_numpy(data["train"]["Z"])
    y = torch.from_numpy(data["train"]["y"])
    name = config.name + (f"_{tag}" if tag else "")
    generator, discriminator = build(config)
    wgan = config.model == "wgan"
    lr, betas = (config.wgan_lr, config.wgan_betas) if wgan else (config.lr, config.betas)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=d_lr or lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(SNAPSHOT_SIZE, config.z_dim, generator=torch.Generator().manual_seed(0))
    fixed_y = torch.arange(SNAPSHOT_SIZE) % 3 if config.conditional else None
    snapshot_dir = SAMPLES_DIR / name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / f"gan_{name}_best.pt"

    history, best_key, step = [], (-1.0, 0.0), 0
    print(f"Training {name}: {len(X):,} real vectors, {config.epochs} epochs, "
          f"{'critic' if wgan else 'discriminator'} lr {d_lr or lr:g}, G lr {lr:g}")
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()
        log = {k: [] for k in ("d_loss", "g_loss", "d_real", "d_fake", "acc_real", "acc_fake",
                               "wasserstein", "gp", "g_grad")}
        for index in torch.randperm(len(X)).split(config.batch_size):
            if len(index) < 2:
                continue
            real, labels = X[index], (y[index] if config.conditional else None)
            z = torch.randn(len(index), config.z_dim)
            step += 1
            sigma = config.instance_noise * (1 - (epoch - 1) / config.epochs)
            noisy = (lambda v: v + sigma * torch.randn_like(v)) if sigma > 0 else (lambda v: v)
            if wgan:
                # ---- critic step (every batch)
                fake = generator(z, labels).detach()
                score_real, score_fake = discriminator(real, labels), discriminator(fake, labels)
                gp = gradient_penalty(discriminator, real, fake, labels)
                d_loss = score_fake.mean() - score_real.mean() + config.gp_lambda * gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                log["d_loss"].append(d_loss.item())
                log["wasserstein"].append((score_real.mean() - score_fake.mean()).item())
                log["gp"].append(gp.item())
                # ---- generator step (every n_critic batches)
                if step % config.n_critic == 0:
                    g_loss = -discriminator(generator(torch.randn(len(index), config.z_dim), labels),
                                            labels).mean()
                    opt_g.zero_grad()
                    g_loss.backward()
                    log["g_grad"].append(grad_norm(generator))
                    opt_g.step()
                    log["g_loss"].append(g_loss.item())
            else:
                fake = generator(z, labels)
                # ---- discriminator step: real → real_label (0.9), fake → 0
                logit_real = discriminator(noisy(real), labels)
                logit_fake = discriminator(noisy(fake.detach()), labels)
                d_loss = (bce(logit_real, torch.full_like(logit_real, config.real_label))
                          + bce(logit_fake, torch.zeros_like(logit_fake)))
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()
                p_real, p_fake = torch.sigmoid(logit_real.detach()), torch.sigmoid(logit_fake.detach())
                log["d_loss"].append(d_loss.item())
                log["d_real"].append(p_real.mean().item())
                log["d_fake"].append(p_fake.mean().item())
                log["acc_real"].append((p_real > 0.5).float().mean().item())
                log["acc_fake"].append((p_fake < 0.5).float().mean().item())
                # ---- generator step, non-saturating: maximise log D(G(z))
                logit = discriminator(noisy(fake), labels)
                g_loss = bce(logit, torch.ones_like(logit))
                opt_g.zero_grad()
                g_loss.backward()
                log["g_grad"].append(grad_norm(generator))
                opt_g.step()
                log["g_loss"].append(g_loss.item())

        row = {"epoch": epoch, **{k: float(np.mean(v)) for k, v in log.items() if v},
               "seconds": round(time.time() - start, 1)}
        if epoch % config.eval_every == 0 or epoch == 1:
            row.update(evaluate(generator, config, data["validation"]["Z"], data["validation"]["y"],
                                data["train"]["y"]))
            with torch.no_grad():
                generator.eval()
                snapshot = generator(fixed_z, fixed_y).numpy().astype(np.float16)
                generator.train()
            np.save(snapshot_dir / f"epoch_{epoch:03d}.npy", snapshot)
            # Highest F1; ties (e.g. F1 = 0 while recall is 0) broken by the lower FD.
            key = (row["val_f1"], -row["val_fd"])
            if key > best_key:
                best_key = key
                torch.save({"generator": generator.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            "config": config.to_dict(), "epoch": epoch, "val_f1": row["val_f1"],
                            "val_fd": row["val_fd"],
                            "d_lr": d_lr or lr, "tag": tag}, best_path)
                row["best"] = True
        history.append(row)
        if wgan:
            line = (f"critic {row['d_loss']:+.3f}  W {row['wasserstein']:.3f}  gp {row['gp']:.3f}  "
                    f"G {row.get('g_loss', float('nan')):+.3f}")
        else:
            line = (f"D {row['d_loss']:.3f}  G {row['g_loss']:.3f}  D(x) {row['d_real']:.2f}  "
                    f"D(G(z)) {row['d_fake']:.2f}  acc real/fake {row['acc_real']:.2f}/{row['acc_fake']:.2f}")
        extra = (f"  | val FD {row['val_fd']:.1f}  P {row['val_precision']:.2f}  R {row['val_recall']:.2f}"
                 f"  F1 {row['val_f1']:.3f}"
                 f"{'  *best' if row.get('best') else ''}") if "val_fd" in row else ""
        print(f"[{name}] epoch {epoch:>3}  {line}  |G grad| {row.get('g_grad', 0):.3f}{extra}"
              f"  ({row['seconds']:.0f}s)", flush=True)

    best_row = max((r for r in history if "val_f1" in r), key=lambda r: (r["val_f1"], -r["val_fd"]))
    result = {"name": name, "config": config.to_dict(), "d_lr": d_lr or lr,
              "best_epoch": best_row["epoch"], "best_val_f1": best_row["val_f1"],
              "best_val_fd": best_row["val_fd"], "best_val_precision": best_row["val_precision"],
              "best_val_recall": best_row["val_recall"],
              "minutes": round((time.time() - start) / 60, 2), "history": history}
    (RUNS_DIR / f"{name}_history.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(RUNS_DIR / f"{name}_log.csv", "w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for r in history for key in r}, key=lambda k: (k != "epoch", k))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    print(f"Done {name} in {result['minutes']} min. Best val F1 {best_row['val_f1']:.3f} (FD "
          f"{best_row['val_fd']:.1f}) at epoch {result['best_epoch']} → {best_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("vanilla", "wgan"), default="vanilla")
    parser.add_argument("--conditional", action="store_true", help="class-conditional (cGAN)")
    parser.add_argument("--epochs", type=int, default=GANConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=GANConfig.batch_size)
    parser.add_argument("--real-label", type=float, default=GANConfig.real_label,
                        help="label of real vectors for the vanilla D (1.0 = no smoothing)")
    parser.add_argument("--d-lr", type=float, default=None, help="override the D/critic learning rate")
    parser.add_argument("--d-dropout", type=float, default=GANConfig.d_dropout,
                        help="dropout in the vanilla discriminator (0 = a stronger D)")
    parser.add_argument("--instance-noise", type=float, default=0.0,
                        help="std of the noise added to D's inputs, annealed to 0 (vanilla)")
    parser.add_argument("--minibatch-std", action="store_true",
                        help="minibatch standard-deviation feature in D (vanilla)")
    parser.add_argument("--seed", type=int, default=GANConfig.seed)
    parser.add_argument("--tag", default="", help="suffix for an ablation run (keeps the main checkpoints)")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    config = GANConfig(model=args.model, conditional=args.conditional, epochs=args.epochs,
                       batch_size=args.batch_size, real_label=args.real_label, seed=args.seed,
                       instance_noise=args.instance_noise, minibatch_std=args.minibatch_std,
                       d_dropout=args.d_dropout)
    train(config, tag=args.tag, d_lr=args.d_lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
