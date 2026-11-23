#!/bin/sh
# Kernel and transitive axiom checks; no production correspondence claim.
set -eu
cd "$(dirname "$0")/.."
python3 scripts/check_proofs.py
cd lean
lake build Pudding Pudding.Extensions
for audit in Audit LangevinAudit AccelerationAudit FlowDataAudit EquilibriumAudit; do
    lake env lean "$audit.lean"
done
