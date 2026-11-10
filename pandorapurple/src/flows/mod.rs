//! Inferno — Forward and backward stochastic drift interfaces and transport-plan latent displacements.

use candle_core::{Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction { Forward, Backward }

/// A drift evaluated at one physical time for the entire batch.
pub trait DriftField {
    fn drift(&self, x: &Tensor, t: &Tensor, direction: Direction) -> Result<Tensor>;
}

/// Latent GEOC barycentric drift. The caller supplies a Sinkhorn plan [N,M]
/// over the current latent batch and target atoms [M,D]. No objective or trainer.
pub fn latent_geoc_drift(x: &Tensor, atoms: &Tensor, plan: &Tensor) -> Result<Tensor> {
    let (n, d) = x.dims2()?;
    let (m, ad) = atoms.dims2()?;
    if ad != d || plan.dims() != [n, m] { candle_core::bail!("GEOC shape mismatch"); }
    let row_mass = plan.sum_keepdim(1)?;
    let min_mass = row_mass.min_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    let total = row_mass.sum_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    let min_weight = plan.min_all()?.to_dtype(candle_core::DType::F64)?.to_scalar::<f64>()?;
    if !min_mass.is_finite() || min_mass <= 0.0 || !total.is_finite() || !min_weight.is_finite() || min_weight < 0.0 {
        candle_core::bail!("GEOC plan requires nonnegative weights and positive finite row mass");
    }
    let mass = row_mass;
    plan.contiguous()?.matmul(&atoms.contiguous()?)?.broadcast_div(&mass)?.sub(x)
}

/// Anderson reverse-time drift b_rev = b_forward - sigma² score.
/// Assumes spatially constant scalar diffusion. Integrate with negative dt.
pub fn anderson_backward_drift(forward: &Tensor, score: &Tensor, sigma: f64) -> Result<Tensor> {
    if !sigma.is_finite() || sigma < 0.0 { candle_core::bail!("invalid diffusion"); }
    forward.sub(&score.affine(sigma * sigma, 0.0)?)
}

/// Latent GEOC geometry for a caller-supplied linear Gaussian reference.
/// C(x,y) = (y - Phi x - offset)^T G^-1 (y - Phi x - offset).
/// Use Sinkhorn temperature 2 with this unscaled cost. Reference construction
/// and any application-specific schedules remain the caller's responsibility.
pub struct LatentGeocDrift {
    phi: Tensor,
    offset: Tensor,
    inverse_gramian: Tensor,
}
impl LatentGeocDrift {
    pub fn new(phi: Tensor, offset: Tensor, inverse_gramian: Tensor) -> Result<Self> {
        let (d, d2) = phi.dims2()?;
        if d == 0 || d != d2 || offset.dims() != [d] || inverse_gramian.dims() != [d,d] {
            candle_core::bail!("invalid latent reference geometry shapes");
        }
        Ok(Self { phi, offset, inverse_gramian })
    }
    fn reference_mean(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.phi.t()?.contiguous()?)?.broadcast_add(&self.offset)
    }
    pub fn ground_cost(&self, x: &Tensor, atoms: &Tensor) -> Result<Tensor> {
        let residual = atoms.unsqueeze(0)?.broadcast_sub(&self.reference_mean(x)?.unsqueeze(1)?)?;
        let (n,m,d) = residual.dims3()?;
        let flat = residual.reshape((n*m,d))?;
        flat.matmul(&self.inverse_gramian.contiguous()?)?.mul(&flat)?.sum(1)?.reshape((n,m))
    }
    /// Forward drift -grad(f)/2 from the entropic plan rows. Anderson reversal
    /// is supplied separately by LangevinSDE using the caller's score.
    pub fn drift(&self, x: &Tensor, atoms: &Tensor, plan: &Tensor) -> Result<Tensor> {
        let displacement = latent_geoc_drift(x, atoms, plan)?;
        let barycenter = displacement.add(x)?;
        barycenter.sub(&self.reference_mean(x)?)?
            .matmul(&self.inverse_gramian.t()?.contiguous()?)?
            .matmul(&self.phi.contiguous()?)
    }
}
