//! Differentiable convolution paths that work with registry Candle.
//!
//! These functions handle non-contiguous inputs and kernels without relying
//! on the historical Candle vendor patches. They materialize gathered patches,
//! so memory use can exceed a native convolution implementation.

use crate::candle::{Result, Tensor};
pub use crate::pandora::model::conv_t::{
    conv1d_diff, conv_transpose1d_diff, forward as forward_conv_transpose1d, forward_conv1d,
};

/// Grouped 2D cross-correlation with differentiable gather/matmul operations.
///
/// Input is `[batch, channels, height, width]`; weights are
/// `[outputs, channels / groups, kernel_height, kernel_width]`.
/// Padding, stride and dilation apply equally to both spatial dimensions.
pub fn conv2d_diff(
    x: &Tensor,
    w: &Tensor,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let (batch, channels, height, width) = x.dims4()?;
    let (outputs, per_group, kh, kw) = w.dims4()?;
    if stride == 0
        || dilation == 0
        || groups == 0
        || kh == 0
        || kw == 0
        || channels == 0
        || outputs == 0
        || channels % groups != 0
        || outputs % groups != 0
        || channels / groups != per_group
    {
        crate::candle::bail!("conv2d_diff: invalid convolution geometry");
    }
    let axis = |length: usize, taps: usize| -> Result<(usize, usize)> {
        let extent = dilation
            .checked_mul(taps - 1)
            .and_then(|n| n.checked_add(1));
        let padded = padding.checked_mul(2).and_then(|n| length.checked_add(n));
        match (extent, padded) {
            (Some(e), Some(p)) if p >= e => Ok((p, (p - e) / stride + 1)),
            _ => crate::candle::bail!(
                "conv2d_diff: kernel extent exceeds input or geometry overflows"
            ),
        }
    };
    let (padded_h, out_h) = axis(height, kh)?;
    let (padded_w, out_w) = axis(width, kw)?;
    let x = if padding == 0 {
        x.clone()
    } else {
        let horizontal = Tensor::zeros((batch, channels, height, padding), x.dtype(), x.device())?;
        let x = Tensor::cat(&[&horizontal, x, &horizontal], 3)?;
        let vertical = Tensor::zeros((batch, channels, padding, padded_w), x.dtype(), x.device())?;
        Tensor::cat(&[&vertical, &x, &vertical], 2)?
    };
    let positions = out_h * out_w;
    let taps = kh * kw;
    let mut indices = Vec::new();
    for row in 0..out_h {
        for col in 0..out_w {
            for kr in 0..kh {
                for kc in 0..kw {
                    indices.push(
                        ((row * stride + kr * dilation) * padded_w + col * stride + kc * dilation)
                            as i64,
                    );
                }
            }
        }
    }
    let indices = Tensor::from_vec(indices, positions * taps, x.device())?;
    let x = x
        .contiguous()?
        .reshape((batch, channels, padded_h * padded_w))?;
    let mut result = Vec::with_capacity(groups);
    for group in 0..groups {
        let patches = x
            .narrow(1, group * per_group, per_group)?
            .contiguous()?
            .index_select(&indices, 2)?
            .reshape((batch, per_group, positions, taps))?
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((batch * positions, per_group * taps))?;
        let weights = w
            .narrow(0, group * (outputs / groups), outputs / groups)?
            .contiguous()?
            .reshape((outputs / groups, per_group * taps))?;
        result.push(
            patches
                .matmul(&weights.t()?)?
                .reshape((batch, out_h, out_w, outputs / groups))?
                .permute((0, 3, 1, 2))?,
        );
    }
    Tensor::cat(&result, 1)?.contiguous()
}
