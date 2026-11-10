# Kernel manifests required for API builds

The checkout omitted these optional path dependencies. Cargo resolves their
manifests even when building only the CPU API. Restored unmodified published
0.11.0 source archives from static.crates.io, verified against crates.io SHA-256:

- candle-kernels: `67450168a281bbb195a14cc85cf164c7a12023a54a03409a3324f476346e0789`
- candle-metal-kernels: `242e83c6acf639bb273c929d73c67a882bb4dd08a140f121096e19ba2f213d3e`

License files are retained in both directories. The API does not enable these
GPU backends; this restores dependency resolution without adding a GPU runtime.
