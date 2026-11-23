# Third-party notices

The crate is MPL-2.0. Imported files retain their upstream notices:

- Autograd scheduling derives from PyTorch; see
  [provenance](../src/autograd/PROVENANCE.md) and
  [BSD notice](../src/autograd/LICENSE.pytorch).
- Native CUDA bridge sources retain the
  [PyTorch BSD notice](../src/cuda/cpp/LICENSE.pytorch) and
  [port map](../src/cuda/PORTED_FROM.md).
- Tensor and autograd sources were previously distributed as dk-tensor and
  dk-autograd under MIT OR Apache-2.0. Their source headers are retained.
- Pudding runtime and proof sources are MPL-2.0; see
  [provenance](../src/pudding/PROVENANCE.md) and [license](../src/pudding/LICENSE).
- Candle is obtained from crates.io under MIT OR Apache-2.0.
  The historical vendor snapshot is retained in the checkout for reference
  and excluded from the published package.
- Lean and mathlib are external tools/dependencies with their own licenses;
  their build artifacts and downloaded dependencies are excluded.
