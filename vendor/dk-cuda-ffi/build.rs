//! First C++ build in this workspace. Everything here is host C++17 — no
//! nvcc; the CUDA toolkit is needed only for headers and shared libraries.
//! Without the `cuda` feature this script is a no-op and the crate compiles
//! pure-Rust stubs, so GPU-less `cargo check/test --workspace` stays green.

use std::path::PathBuf;

fn find_cuda() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(p) = std::env::var(var) {
            let p = PathBuf::from(p);
            if p.join("include/cuda.h").exists() {
                return p;
            }
        }
    }
    let default = PathBuf::from("/usr/local/cuda");
    if default.join("include/cuda.h").exists() {
        return default;
    }
    // nvcc's parent-parent as a last resort
    if let Ok(out) = std::process::Command::new("which").arg("nvcc").output() {
        if out.status.success() {
            let nvcc = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Some(root) = PathBuf::from(nvcc).parent().and_then(|p| p.parent()) {
                if root.join("include/cuda.h").exists() {
                    return root.to_path_buf();
                }
            }
        }
    }
    panic!(
        "dk-cuda-ffi built with the `cuda` feature but no CUDA toolkit found. \
         Set CUDA_HOME (or CUDA_PATH) to the toolkit root, or build inside the \
         nvidia/cuda *-devel image (see Dockerfile.worker)."
    );
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=src/bridge.rs");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let cuda = find_cuda();
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());

    cxx_build::bridge("src/bridge.rs")
        .file("cpp/ping.cpp")
        .file("cpp/pqo3.cpp")
        .include(&manifest)
        .include(cuda.join("include"))
        .std("c++17")
        .flag_if_supported("-fno-strict-aliasing")
        .compile("dk-cuda-ffi-cpp");

    // Library search paths: lib64 for the real libs, stubs for libcuda at
    // link time (the real driver resolves at runtime).
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64/stubs").display()
    );
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");
    println!("cargo:rustc-link-lib=cudnn");
}
