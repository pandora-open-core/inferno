//! Small, backend-independent consistency obligations over host snapshots.

/// An owned observation. Values are in flattened row-major order.
/// Invalid observations are retained so the engine can report their failure.
#[derive(Clone, Debug)]
pub struct TensorSnapshot {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}
impl TensorSnapshot {
    pub fn new(shape: impl Into<Vec<usize>>, values: Vec<f64>) -> Self {
        Self {
            shape: shape.into(),
            values,
        }
    }
    fn valid(&self) -> bool {
        self.shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) == Some(self.values.len())
            && self.values.iter().all(|v| v.is_finite())
    }
}

/// Elementwise comparison: |a-b| <= absolute + relative * max(|a|, |b|).
/// Both tolerances must be finite and nonnegative. No defaults are inferred
/// from a workflow's precision or conditioning.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl Tolerance {
    pub const fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }
    fn valid(self) -> bool {
        self.absolute.is_finite()
            && self.relative.is_finite()
            && self.absolute >= 0.0
            && self.relative >= 0.0
    }
    fn accepts(self, a: f64, b: f64) -> bool {
        if !self.valid() || !a.is_finite() || !b.is_finite() {
            return false;
        }
        // Scale before subtraction to avoid accepting inf <= inf on overflow.
        let scale = a.abs().max(b.abs()).max(self.absolute);
        if scale == 0.0 {
            return true;
        }
        (a / scale - b / scale).abs()
            <= self.absolute / scale + self.relative * (a.abs().max(b.abs()) / scale)
    }
}

/// A bounded numerical obligation, with no claim about unobserved inputs.
pub enum Check<'a> {
    Finite(&'a TensorSnapshot),
    Shape {
        tensor: &'a TensorSnapshot,
        expected: &'a [usize],
    },
    Close {
        actual: &'a TensorSnapshot,
        expected: &'a TensorSnapshot,
        tolerance: Tolerance,
    },
    /// Check a nonempty rank-one probability vector, including nonnegativity.
    Probability {
        tensor: &'a TensorSnapshot,
        tolerance: Tolerance,
    },
}
#[derive(Clone, Debug)]
pub struct CheckResult {
    pub label: String,
    pub passed: bool,
    pub detail: &'static str,
}
/// Runtime evidence only. An empty report is deliberately inconclusive.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub checks: Vec<CheckResult>,
}
impl Report {
    pub fn passed(&self) -> bool {
        !self.checks.is_empty() && self.checks.iter().all(|c| c.passed)
    }
}
#[derive(Default)]
pub struct Engine {
    report: Report,
}
impl Engine {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn check(&mut self, label: impl Into<String>, check: Check<'_>) -> &mut Self {
        let (passed, detail) = match check {
            Check::Finite(t) => (t.valid(), "shape/storage agreement and finite values"),
            Check::Shape { tensor, expected } => (
                tensor.valid() && tensor.shape == expected,
                "exact shape and valid storage",
            ),
            Check::Close {
                actual,
                expected,
                tolerance,
            } => (
                tolerance.valid()
                    && actual.valid()
                    && expected.valid()
                    && actual.shape == expected.shape
                    && !actual.values.is_empty()
                    && actual
                        .values
                        .iter()
                        .zip(&expected.values)
                        .all(|(&a, &b)| tolerance.accepts(a, b)),
                "nonempty elementwise agreement within explicit tolerance",
            ),
            Check::Probability { tensor, tolerance } => {
                let sum: f64 = tensor.values.iter().sum();
                (
                    tensor.valid()
                        && tensor.shape.len() == 1
                        && !tensor.values.is_empty()
                        && tensor.values.iter().all(|v| *v >= 0.0)
                        && tolerance.accepts(sum, 1.0),
                    "nonempty rank-one nonnegative vector with sum near one",
                )
            }
        };
        self.report.checks.push(CheckResult {
            label: label.into(),
            passed,
            detail,
        });
        self
    }
    pub fn finish(self) -> Report {
        self.report
    }
}
