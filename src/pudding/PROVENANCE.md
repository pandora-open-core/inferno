# Provenance

Pudding is created and contributed by Liam Dziga Douglas Giles, founder and CEO
of Pandora-AI Proprietary Ltd. The public source is offered under MPL-2.0.
External Lean and mathlib dependencies retain their own licenses.

The numerical evaluator and formal package derive from the dk-lean4 development
at source revision `d1ba8f60d15e81bf0b1c0eccd84a7d1cb77679ae`, with subsequent
finite-support equilibrium extensions. This identifies lineage, not an exact
byte-for-byte correspondence to that revision.

The maintained formal source is in `proofs/lean`, with its Lean toolchain and
mathlib revision pinned by the package files. Research plans, historical run
reports, duplicate implementation sources and machine-specific deployment
recipes are excluded from this SDK. Validation must be performed against the
current source; historical runs are not evidence for this release.

This copy was incorporated into inferno-rs from the newer pudding-open-core
checkout (source edits through 2026-11-10). The runtime lives in src/pudding
and formal sources in proofs/. The Inferno adapter consumes this module
without a sibling package or registry dependency on pudding.
