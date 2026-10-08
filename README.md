# Shape-Bayes

Closed-form PCA shape posterior from arXiv 2610.09032 (Mani Kumar Tellamekala, Tosh Brown, Michel Valstar).

A 2D shape with N landmarks is stored as a vector of length 2N. Coordinates are flattened in row-major order: x0, y0, x1, y1, and so on through the last landmark. The shape manifold is S = Sbar + P b, where Sbar has shape (2N,), P has shape (2N, K), and b has shape (K,).

The observation model is a diagonal Gaussian. The two-column PDF places a superscript 2 on the line above the epsilon, so the precision is W = diag(1 / (sigma^2 + eps)), not 1 / (sigma + eps). sigma is the base model's per-coordinate aleatoric uncertainty and has the same shape as S. eps is a small positive constant. Tests use 1e-6.

The conditional prior on b is zero-mean with precision Lambda_prior = diag(exp(logits)), positive by construction. The posterior precision is P transposed times diag(W) times P, plus Lambda_prior. The posterior mean b_mu is the solution of that linear system against P transposed times W times (S' - Sbar). The code uses a linear solve and does not invert the precision explicitly. The posterior shape is S_post = Sbar + P b_mu.

The uncertainty target is the root-mean-square deviation of the teacher shapes from the unaugmented target, coordinate by coordinate. It is not the standard deviation around the ensemble mean. The unaugmented prediction is the target. If that target is not the mean of the teachers, the RMS target and the ensemble standard deviation differ.

The training loss is a weighted sum of the squared Euclidean error between b_mu and b_gt and a Gaussian negative log-likelihood in the coefficient space. Gradients through b_mu are stopped inside the NLL only. The squared-error term still depends on b_mu. The log-determinant comes from slogdet.

E_phi is a minimal stand-in. Each landmark is a token [x, y, sigma_x, sigma_y]. One block embeds those tokens to 32 dimensions, applies 2-head self-attention with a residual, mean-pools over landmarks, and maps the pool to K logits. Parameters are a pytree initialized from a PRNG key.

## Install

```bash
pip install -e .
```

## Run

```bash
JAX_PLATFORMS=cpu python -m pytest -q
```

H200 smoke is `train.sbatch` plus `smoke_gpu.py`: 20 Adam steps of the shape posterior loss on a synthetic batch. Job 759286 on compute-gpu-01, jax 0.11.1 CudaDevice(id=0), loss -0.199159 to -0.322757, exit 0.

What does not match a full experiment: no 300W, COFW, or WFLW images, and no base landmark detector. Callers pass S' and sigma'. The transformer is a minimal stand-in for E_phi, not the unpublished architecture. There are no NME numbers. The in-distribution Mahalanobis check is not included. Precision uses 1/(sigma^2+eps).
