//! dk-cuda-ffi — cxx bridge over ATen-derived cuBLAS/cuDNN C++.
//!
//! The C++ under `cpp/` is transcribed from pytorch @ 540f3a53 (v2.14.0a0),
//! BSD-3-Clause — see `cpp/LICENSE.pytorch` and `PORTED_FROM.md` for the
//! function-by-function provenance map. c10 is not vendored; the small shim
//! layer in `cpp/shim/` replaces it (see the headers there for the exact
//! substitution table).
//!
//! Without the `cuda` feature every entry point returns
//! `Err(CudaUnavailable)` and no C++ is compiled.

#[cfg(feature = "cuda")]
mod bridge;

#[derive(Debug)]
pub enum FfiError {
    CudaUnavailable,
    Cpp(String),
    InvalidInput(&'static str),
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FfiError::CudaUnavailable => {
                write!(f, "dk-cuda-ffi built without the `cuda` feature")
            }
            FfiError::Cpp(m) => write!(f, "{m}"),
            FfiError::InvalidInput(m) => write!(f, "{m}"),
        }
    }
}
impl std::error::Error for FfiError {}

pub type Result<T> = std::result::Result<T, FfiError>;

#[cfg(feature = "cuda")]
impl From<cxx::Exception> for FfiError {
    fn from(e: cxx::Exception) -> Self {
        FfiError::Cpp(e.what().to_string())
    }
}

/// CUDA runtime version (e.g. 12040). The Phase-0 compile/link smoke test.
pub fn ping() -> Result<i32> {
    #[cfg(feature = "cuda")]
    {
        Ok(bridge::ffi::ping()?)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(FfiError::CudaUnavailable)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn ping_stub_errors_without_cuda() {
        assert!(super::ping().is_err());
    }

    /// Proves the cxx bridge links against libcudart (and that the shim
    /// headers compile against the real cuBLAS/cuDNN headers). Needs the
    /// toolkit, NOT a GPU — cudaRuntimeGetVersion is device-free.
    #[test]
    #[cfg(feature = "cuda")]
    fn cudart_links_and_reports_a_version() {
        let v = super::ping().expect("cudaRuntimeGetVersion failed");
        assert!(v >= 11000, "implausible CUDA runtime version {v}");
    }
}

/// Single-query PQO3 device geometry. Pointers and stream are CUDA addresses,
/// not host pointers. Each page is [K/V, page_tokens, kv_heads, dim], 14 bytes
/// per 32 values. A launch synchronizes its stream before returning.
#[derive(Debug, Clone, Copy)]
pub struct Pqo3Launch {
    pub query: u64,
    pub pages: u64,
    pub output: u64,
    pub stream: u64,
    pub heads: u32,
    pub kv_heads: u32,
    pub dim: u32,
    pub page_tokens: u32,
    pub page_count: u32,
    pub tokens: u32,
}
impl Pqo3Launch {
    #[cfg(any(feature = "cuda", test))]
    fn geometry(self) -> Result<(usize, usize)> {
        if self.heads == 0
            || self.heads > 65536
            || self.kv_heads == 0
            || self.heads % self.kv_heads != 0
            || ![32, 64, 128, 256].contains(&self.dim)
            || self.page_tokens == 0
            || self.page_count == 0
            || self.page_count > 256
            || self.tokens == 0
            || u64::from(self.tokens) > u64::from(self.page_tokens) * u64::from(self.page_count)
        {
            return Err(FfiError::InvalidInput("invalid PQO3 geometry"));
        }
        let elements = u64::from(self.heads) * u64::from(self.dim);
        let packed = 2u64
            .checked_mul(u64::from(self.page_tokens))
            .and_then(|n| n.checked_mul(u64::from(self.kv_heads)))
            .and_then(|n| n.checked_mul(u64::from(self.dim) / 32))
            .and_then(|n| n.checked_mul(14))
            .and_then(|n| n.checked_mul(u64::from(self.page_count)))
            .filter(|n| *n <= 256 * 1024 * 1024)
            .ok_or(FfiError::InvalidInput("PQO3 packed buffer exceeds bound"))?;
        Ok((elements as usize, packed as usize))
    }
}
#[cfg(any(feature = "cuda", test))]
fn image_bounds(image: &[u8]) -> Result<()> {
    // KeOps prepare produces CUBIN, not unterminated PTX read by a driver C API.
    if image.len() > 64 * 1024 * 1024 || !image.starts_with(b"\x7fELF") {
        return Err(FfiError::InvalidInput(
            "expected a bounded trusted CUBIN image",
        ));
    }
    Ok(())
}
/// Device architecture and driver version used by the model's compile cache.
pub fn device_fingerprint(_device: i32) -> Result<String> {
    #[cfg(feature = "cuda")]
    {
        Ok(bridge::ffi::device_fingerprint(_device)?)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(FfiError::CudaUnavailable)
    }
}
/// Launch the model's fused single-query attention and synchronize its stream.
///
/// # Safety
/// `image` must be a complete, trusted compiler-produced CUBIN implementing
/// dk_pqo3_decode_attention. The calling thread must bind its matching CUDA
/// context. All device addresses and page-table entries must have the declared
/// extents, belong to that context, remain alive, and not alias output. Stream
/// ownership remains with the caller. Never accept an untrusted executable image.
pub unsafe fn launch_pqo3(_image: &[u8], _p: Pqo3Launch) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        _p.geometry()?;
        image_bounds(_image)?;
        if _p.query == 0 || _p.pages == 0 || _p.output == 0 {
            return Err(FfiError::InvalidInput("null PQO3 device address"));
        }
        unsafe {
            bridge::ffi::launch_pqo3(
                _image,
                _p.query,
                _p.pages,
                _p.output,
                _p.stream,
                _p.heads,
                _p.kv_heads,
                _p.dim,
                _p.page_tokens,
                _p.page_count,
                _p.tokens,
            )?;
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(FfiError::CudaUnavailable)
    }
}
/// Own temporary device buffers, bind/restore the selected primary context,
/// launch synchronously, and return host output. Address/stream fields in `p`
/// are intentionally ignored; only its geometry applies to host-buffer calls.
///
/// # Safety
/// `image` must be a complete trusted CUBIN implementing the documented kernel
/// ABI. Host bounds validation cannot make arbitrary GPU executable code safe.
pub unsafe fn pqo3_host(
    _image: &[u8],
    _query: &[f32],
    _packed: &[u8],
    _p: Pqo3Launch,
    _device: i32,
) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        let (elements, bytes) = _p.geometry()?;
        image_bounds(_image)?;
        if _device < 0
            || _query.len() != elements
            || _packed.len() != bytes
            || _query.iter().any(|v| !v.is_finite())
        {
            return Err(FfiError::InvalidInput(
                "PQO3 host input bounds or nonfinite query",
            ));
        }
        let output = unsafe {
            bridge::ffi::pqo3_host(
                _image,
                _query,
                _packed,
                _p.heads,
                _p.kv_heads,
                _p.dim,
                _p.page_tokens,
                _p.page_count,
                _p.tokens,
                _device,
            )?
        };
        if output.len() != elements || output.iter().any(|v| !v.is_finite()) {
            return Err(FfiError::InvalidInput("invalid PQO3 device output"));
        }
        Ok(output)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(FfiError::CudaUnavailable)
    }
}
#[cfg(test)]
mod attention_tests {
    use super::*;
    fn geometry() -> Pqo3Launch {
        Pqo3Launch {
            query: 0,
            pages: 0,
            output: 0,
            stream: 0,
            heads: 8,
            kv_heads: 2,
            dim: 64,
            page_tokens: 16,
            page_count: 2,
            tokens: 31,
        }
    }
    #[test]
    fn packed_geometry_matches_page_abi_and_rejects_oversized_or_invalid_input() {
        let p = geometry();
        assert_eq!(
            p.geometry().unwrap(),
            (512, 2 * 16 * 2 * (64 / 32) * 14 * 2)
        );
        for bad in [
            Pqo3Launch { heads: 7, ..p },
            Pqo3Launch { dim: 65, ..p },
            Pqo3Launch { tokens: 33, ..p },
            Pqo3Launch {
                page_count: 257,
                ..p
            },
            Pqo3Launch {
                page_tokens: u32::MAX,
                ..p
            },
            Pqo3Launch {
                heads: 65536,
                kv_heads: 65536,
                page_tokens: u32::MAX,
                page_count: 256,
                ..p
            },
        ] {
            assert!(bad.geometry().is_err());
        }
        assert!(image_bounds(b"not cubin").is_err());
    }
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn every_attention_entry_point_fails_explicitly_without_cuda() {
        assert!(matches!(
            device_fingerprint(0),
            Err(FfiError::CudaUnavailable)
        ));
        unsafe {
            assert!(matches!(
                launch_pqo3(&[], geometry()),
                Err(FfiError::CudaUnavailable)
            ));
            assert!(matches!(
                pqo3_host(&[], &[], &[], geometry(), 0),
                Err(FfiError::CudaUnavailable)
            ));
        }
    }
}
