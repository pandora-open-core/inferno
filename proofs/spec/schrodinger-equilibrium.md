# Schrödinger equilibrium on finite atom support

## State and Langevin convention

Freeze support, reward, prior, geometry and all model/observer parameters. An atom
index labels a finite support point in evidence coordinates or in an explicitly
chosen atom chart. These are not automatically the model's simplex coordinates.
Supply positive symmetric graph conductances w_ij. For diffusion D>0 set

    U_i = log pi0_i + beta R_i/2
    mu_i = exp(U_i)/sum exp(U)
    Q_ij = D w_ij exp((U_j-U_i)/2), i != j
    Q_ii = -sum_(j != i) Q_ij
    K = I + dt Q, 0 < dt < 1/max_i(-Q_ii).

Then mu_i Q_ij = mu_j Q_ji. This is a continuous-time finite graph jump generator
and a lazy uniformisation kernel, not the Euler discretisation of a continuous
SDE. With suitable local mesh conductances it is a Langevin discretisation;
no mesh-to-diffusion convergence theorem is supplied. Its invariant mass is exact
in real arithmetic at each admissible step. The continuous convention is
b=D grad U, noise sqrt(2D), so the candidate is pi0 exp(beta R/2).
Conductance is an explicit geometry input; changing it does not change mu when
symmetry is retained. The implementation currently requires complete support,
positive masses and strictly positive diagonal, avoiding unsupported feasibility
and periodicity claims. Floating-point underflow/overflow is rejected.

## Schrödinger endpoint condition and equilibrium

Use the actual reference endpoint joint R_ij=mu_i K_ij. Solve

    minimise KL(P | R), P 1 = a, P^T 1 = b
    P_ij = u_i R_ij v_j, T_ij = P_ij/a_i.

No arbitrary squared-distance replacement for the reference kernel is made.
The KL coefficient is one for this fixed reference; beta, D, dt and an independent
Sinkhorn epsilon are not identified. This is a single reference step; a longer
horizon would require a declared K power or exp(hQ). It does not construct a
continuous Brownian path bridge.

The equilibrium condition is a=b=rho, exactly balanced P, and T=P/rho:
rho T=rho. If R is symmetric, the unique positive finite KL minimiser is symmetric
by transpose invariance and strict convexity. It can be written diag(s) R diag(s),
so rho_i T_ij=rho_j T_ji. For rho=mu the reference itself is the solution and T=K.
For unequal endpoints T transports a to b and a generally is not stationary.
The equal-endpoint stationary chain is a homogeneous chain derived from the
endpoint coupling; do not identify it with every intermediate law of a
finite-horizon Schrödinger path bridge.

Lean proves detailed-balance-to-stationarity, symmetry of the symmetric scaling,
nonnegative stochastic conditional rows and endpoint transport from feasibility,
and equilibrium/reversibility from feasible symmetric equal-endpoint couplings.
These are exact conditional finite theorems. The Lean development does not yet
prove factor existence, KL optimality/uniqueness, continuous SDE existence, or
convergence of the Rust Sinkhorn implementation. No axioms are added.

## Numerical and production boundary

`src/equilibrium.rs` provides the reference and alternating endpoint scaling.
Both marginal residuals must pass the caller's tolerance; exhaustion and invalid
numeric domains return errors. Conditional row error can be as large as absolute
marginal tolerance divided by the smallest source mass. Binary64 tests do not
constitute an interval certificate or a Lean proof about executed Rust.

The conditional rows retain stochastic choices: a consumer can sample an atom
index directly. A deterministic pushforward preserves a stationary chain's
one-time law but need not produce a Markov chain. Integration with a learned
model must explicitly bind support, observation ordering, shape, floor bias,
detach semantics and RNG state. No learned-model integration is certified here.
