# Team 4 — Workshop #5: Generative Adversarial Networks for Review Embeddings

Machine Learning (Semester 2026-III) — Systems Engineering,
**Universidad Distrital Francisco José de Caldas**. Instructor: Eng. Carlos Andrés Sierra, M.Sc.

Team: Xiomara Salome Arias Arias · Julian David Celis Giraldo · Alejandro Gomez Moyano

A vanilla GAN and a WGAN-GP (plus their class-conditional versions) learn the distribution of
the 42,632 Disneyland reviews in the space of Workshop #2's fine-tuned RoBERTa CLS vectors.
The synthetic vectors are evaluated for quality, diversity and memorisation, then used to
augment the training set of Workshop #2's classifier head.

**Why embeddings, not text.** Words are discrete tokens: picking one is an argmax with no
gradient, so the discriminator's signal cannot reach a text generator without reinforcement
learning (SeqGAN) or relaxations (Gumbel-softmax). The 768-d CLS vectors are continuous,
tabular-like data — the brief's "tabular" case: fully-connected generator and discriminator.
This is also the space of Workshop #4's VAE, so the two generative families compare directly.

## Repository structure

```
src/
├── config.py              paths, labels and every GAN hyperparameter (GANConfig)
├── data.py                CLS vectors + frozen split, StandardScaler, Workshop #2 head
├── gan.py                 Generator, Discriminator, Critic, gradient penalty, sampling
├── train_gan.py           training loop: vanilla GAN (BCE) and WGAN-GP, conditional or not
├── evaluate_gan.py        generation-quality metrics (FD, classifier score, precision/recall, …)
├── generate_samples.py    sample grids, PCA maps, training evolution, training curves
└── augment_and_retrain.py data-augmentation experiment on Workshop #2's head
checkpoints/
├── gan_vanilla_best.pt        best vanilla generator (+ discriminator)
├── gan_wgan_best.pt           best WGAN-GP generator (+ critic)
├── gan_vanilla_cond_best.pt   best conditional vanilla GAN
├── gan_wgan_cond_best.pt      best conditional WGAN-GP
└── w2_head.pt                 Workshop #2's classifier head (cut out of its 499 MB checkpoint)
runs/workshop5/
├── <model>_history.json / <model>_log.csv   per-epoch training logs
├── samples/<model>/epoch_XXX.npy            64 vectors from a fixed noise batch, every 10 epochs
├── evaluation.json / evaluation.md          generation-quality metrics
├── augmentation.json / augmentation.md      augmentation experiment
├── samples_nearest_reviews.md               nearest real review to generated vectors
└── figures/                                 every figure of the report
```

## Setup

Python 3.12–3.14 and Poetry. The networks are small MLPs: the CPU build of torch trains each
model in about 20 minutes (no GPU needed).

```bash
poetry install
```

## Data

Reuses the cleaned corpus and frozen split of Workshop #1, the RoBERTa CLS vectors extracted
in Workshop #4, and the fine-tuned checkpoint of Workshop #2 (only its head is used):

```bash
mkdir -p data/silver data/gold checkpoints
cp <workshop1>/data/silver/reviews.parquet         data/silver/
cp <workshop1>/data/gold/splits.json               data/gold/
cp <workshop4>/data/gold/embeddings_bert.npy       data/gold/
cp <workshop4>/data/gold/labels.npy                data/gold/
cp <workshop2>/checkpoints/model_best.pt           checkpoints/   # only if w2_head.pt is missing
```

## Reproduce

```bash
poetry run python src/train_gan.py --model vanilla
poetry run python src/train_gan.py --model wgan
poetry run python src/train_gan.py --model vanilla --conditional
poetry run python src/train_gan.py --model wgan --conditional
# failure-mode ablations (vanilla GAN)
poetry run python src/train_gan.py --model vanilla --minibatch-std --tag mbstd
poetry run python src/train_gan.py --model vanilla --instance-noise 0.5 --tag noise
poetry run python src/train_gan.py --model vanilla --d-dropout 0 --d-lr 4e-4 --tag strongd
poetry run python src/train_gan.py --model vanilla --real-label 1.0 --d-dropout 0 --d-lr 1e-3 --epochs 60 --tag unstable

poetry run python src/evaluate_gan.py
poetry run python src/generate_samples.py
poetry run python src/augment_and_retrain.py
```

## Models

| | Vanilla GAN | WGAN-GP |
| --- | --- | --- |
| Generator | z (128) → 256 → 512 → 1024 → 768; BatchNorm + ReLU; linear output | same |
| Second network | Discriminator 768 → 512 → 256 → 1; LeakyReLU 0.2, dropout 0.3, Sigmoid (fused into BCEWithLogits) | Critic, same layers, LayerNorm instead of dropout, no Sigmoid |
| Loss | BCE; G non-saturating −log D(G(z)); real labels 0.9 | C: E[C(x̃)] − E[C(x)] + 10·GP; G: −E[C(G(z))] |
| Optimiser | Adam 2e-4, β = (0.5, 0.999), both nets | Adam 1e-4, β = (0, 0.9), 5 critic steps per G step |
| Training | 150 epochs, batch 256 | 150 epochs, batch 256 |

Conditional versions concatenate a 16-d learned label embedding to z and to D's input.
The output is linear because the targets are standardised vectors (mean 0, std 1 per
dimension, unbounded): a Tanh would clip their tails.

**Checkpoint selection.** Every 10 epochs, 5,000 generated vectors are compared with the real
validation split; the checkpoint kept has the highest *generative F1* (harmonic mean of
precision and recall), ties broken by the lower Fréchet distance. FD alone is not enough here:
a plain Gaussian fitted to the training vectors reaches the real-vs-real FD (5.9 vs. 5.5) while
only 10 % of its samples are realistic (precision 0.10).

## Results

Generation quality on the held-out test split, 10,000 generated vectors per model
(full table: `runs/workshop5/evaluation.md`):

| Source | FD ↓ | Precision ↑ | Recall ↑ | Label F1 (cGAN) |
| --- | --- | --- | --- | --- |
| Real (held-out) | 5.8 | 0.748 | 0.717 | 0.735 |
| Vanilla GAN | 198.7 | 0.731 | 0.000 | — |
| WGAN-GP | 35.3 | 0.555 | 0.611 | — |
| cGAN (vanilla) | 233.7 | 0.716 | 0.000 | 0.839 |
| cGAN (WGAN-GP) | **28.0** | 0.566 | 0.585 | **0.875** |

The vanilla GAN collapses onto the dense positive region (recall ≈ 0); minibatch
discrimination, instance noise and a stronger discriminator do not fix it, WGAN-GP does.

Augmentation of Workshop #2's head, test macro-F1 over 3 seeds (full table:
`runs/workshop5/augmentation.md`): real only 0.7127 ± 0.0027; best +5,000 cGAN (vanilla)
vectors 0.7196 ± 0.0021 (+0.007); +5,000 cGAN (WGAN-GP) 0.7186 ± 0.0016. Gains are small
(≤ 0.7 points) and come mostly from conditioning (correct labels), not from sample diversity.

The report draft is `report/workshop_5/group4_workshop5_report.md`.
