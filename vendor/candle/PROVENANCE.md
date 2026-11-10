# Candle vendor snapshot

The Candle 0.11.0 vendor snapshot was imported on 2026-11-06. Core and NN retain the source
workspace's local convolution fixes, including contiguous kernels and unsigned
transposed-convolution output-length arithmetic. Both are version 0.11.0.
Core, NN, CUDA kernels and Metal kernels are copied without source changes.
`source-manifest.json` records SHA-256 hashes of all 269 copied crate files.
The workspace member list is reduced to core and NN; dependency definitions
are retained. Kernel archive provenance is in RESTORED_KERNELS.md.

Inferno uses workspace-inherited path dependencies for Candle core and NN.
Consumers do not need patch tables. Pudding has no Candle dependency; Inferno
owns the diagnostic adapter. Native conv-transpose1d backward remains
unsupported; the differentiable composition is the decoder path.
