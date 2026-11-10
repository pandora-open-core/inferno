# Contributing

Inferno is an alpha SDK. Keep changes focused, document public API behavior and
include regression coverage for numerical or gradient changes. Contributions
are provided under MPL-2.0. Preserve third-party notices and licenses.

Use the sibling Pudding checkout described in the README. Before submitting:

```sh
cargo fmt --package inferno-rs -- --check
python3 tools/check_public_scope.py
python3 tools/check_consumer.py
cargo test --workspace --locked
cargo doc -p inferno-rs --no-deps --locked
```

State the tensor backend, dtype, device, tolerances and expected shapes for
numerical changes. GPU changes require hardware-specific validation. Update
`tools/public_modules.json` when intentionally changing the component inventory.
Keep model weights, private configuration, credentials and generated artifacts
out of contributions.
