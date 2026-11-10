# Changelog

## 0.1.0 — alpha

- Rename the curated `rustysecrets` crate and public module to `pandorapurple`.

- Provide tensors, autograd, neural model components and Gaussian-process tools.
- Expose the kernel-network multitask GP as `models::PandoraGP` and the existing generic spectral VAE as `models::PandoraVae`.
- Integrate Pudding runtime checks through `pudding` and `diagnostics`.
- Remove research planning documents and separate experimental model wrappers.

- Remove generated reduction backends, KeOps support and the Cloud client/runbook.
- Retain the legacy neural kernel; keep the newer protected coupling kernel excluded.
- Complete CPU covariance families and add differentiable Candle standard kernels.
- Replace module captions with library-focused descriptions.

This source alpha has no bundled model weights and no stable API guarantee.
