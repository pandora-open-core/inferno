//! The cxx bridge. Compiled only with the `cuda` feature; the bridge module
//! grows per phase (Phase 0: ping; Phase 5: blas/conv/batchnorm surfaces).
//!
//! Conventions (from the port plan):
//! - device pointers cross as raw pointers in `unsafe extern "C++"`, or as
//!   `usize` inside shared structs (cxx shared structs cannot hold pointers);
//! - streams cross as `usize` (cudarc raw stream);
//! - all fallible C++ throws -> `Result` here.

#[cxx::bridge(namespace = "dk::cuda")]
pub mod ffi {
    unsafe extern "C++" {
        include!("cpp/ping.h");
        include!("cpp/pqo3.h");
        fn device_fingerprint(device: i32) -> Result<String>;
        unsafe fn launch_pqo3(
            image: &[u8],
            query: u64,
            pages: u64,
            output: u64,
            stream: u64,
            heads: u32,
            kv_heads: u32,
            dim: u32,
            page_tokens: u32,
            page_count: u32,
            tokens: u32,
        ) -> Result<()>;
        unsafe fn pqo3_host(
            image: &[u8],
            query: &[f32],
            packed: &[u8],
            heads: u32,
            kv_heads: u32,
            dim: u32,
            page_tokens: u32,
            page_count: u32,
            tokens: u32,
            device: i32,
        ) -> Result<Vec<f32>>;

        /// CUDA runtime version, or Err if the runtime is unusable.
        /// Phase-0 smoke test for compile + link.
        fn ping() -> Result<i32>;
    }
}
