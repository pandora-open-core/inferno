//! Element types. Deliberately small: the curated op set needs exactly these
//! four, and complex is out of scope (see the port plan — no curated op uses
//! it, so the ATen complex instantiations were dropped rather than shimmed).

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 | DType::BF16 => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
