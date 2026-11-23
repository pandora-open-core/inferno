# crates.io release preparation

`inferno-rs` is a single package. Its Rust library name is `inferno`:

```toml
[dependencies]
inferno = { package = "inferno-rs", version = "0.1.0" }
```

The crate includes `tensor`, `autograd`, `pandora`, `cuda`, and `pudding` modules.
Existing `inferno::pandorapurple`, root tensor types, `Variable`, `model`,
`flows`, and `stochastics` paths remain available. Candle core/nn 0.11.1
are registry dependencies; there are no path dependencies or patch tables.
The historical Candle vendor snapshot stays outside the release archive.

Pudding was copied from the newer `pudding-open-core` distribution, including
its MPL-2.0 provenance, runtime tests, numerical reference, and Lean proofs.
The older `pudding/research/dk-lean4` crate declares a proprietary license and
was not incorporated. Lean itself remains an external proof checker; normal
Cargo builds require neither Lean nor mathlib. Proof source and pinned
configuration are packaged; downloaded dependencies and compiled proofs are not.
See [the proof contract](../proofs/README.md) and [architecture](ARCHITECTURE.md)
for the distinction between runtime checks and formal results.

Local checks:

```sh
cargo fmt --all -- --check
python3 tools/check_public_scope.py
cargo check --all-targets --locked
cargo test --locked --features torch-kernels
python3 tools/check_consumer.py
cargo doc --no-deps --locked
cargo package --locked --allow-dirty
```

The package command compiles an unpacked archive. It does not upload it.
The `--allow-dirty` flag permits review of this uncommitted migration.
Review and commit the changes before doing a final release check without that
flag. No publish, push, tag, or commit is part of this migration.

Optional native GPU builds require their own CUDA/cuDNN environment. The
`cuda` feature enables tensor/Candle GPU support, `torch-kernels` exposes the
bridge alias with CPU stubs, and `torch-kernels-cuda` builds the C++ bridge.
