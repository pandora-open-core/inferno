# Pudding proofs

The Lean library is `Pudding` (Lake package `puddingProofs`), renamed from
`DkLean4`. Import `Pudding` for the core or `Pudding.Extensions` for the additional
normalisation, mass truncation, data-flow and map-certificate statements.

The corpus keeps the mathematical ingredients for gradient checking and a
finite-support Schrödinger bridge under the declared Langevin convention:

| Modules | Role |
| --- | --- |
| `Pudding.Autograd` | Local derivatives, composition, truncation and conditional implicit Sinkhorn adjoints |
| `Pudding.Symbolic` | Expression semantics, rewrites and bytecode evaluation identities |
| `Pudding.Langevin` | Drift/density identities and conditional finite-support equilibrium and reversibility |
| `Pudding.Transport` | Finite transport, scaling updates, residuals, gauges and conditional existence/convergence results |
| `Pudding.Variational`, `Pudding.Mean` | Objective, estimator, stationarity and integration-by-parts ingredients |
| `Pudding.Amortised` | Carrier, readout, state layout and conditional error budgets |

Small algebraic lemmas remain where they support these obligations. The standalone
rational stick-breaking example (`Step9Rational`), cache admission/reuse model
(`Cache`) and unused root state wrappers were removed because they did not
contribute to this proof chain. The exact two-atom transport example remains as
a concrete check of the coupling and scaling definitions.

These derivative statements are ingredients for a gradient checker; they do not
certify an executable automatic differentiation implementation. The Langevin
results do not establish continuous SDE bridge existence. See the
[finite-support contract](spec/schrodinger-equilibrium.md) for hypotheses and limits.

Run the source gate, kernel build and transitive axiom audits with the pinned
Lean/mathlib environment:

```sh
sh proofs/scripts/check-lean.sh
```

`Audit.lean` imports `Pudding.Extensions` and checks the retained library's
transitive axiom closure. The focused audit files provide additional diagnostics
for the Langevin and equilibrium results.
