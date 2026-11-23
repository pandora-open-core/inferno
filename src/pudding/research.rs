//! Reproducible finite-support experiments and portable visual evidence.
//! Numerical observations never confer Lean certification on a solver run.
use crate::pudding::equilibrium::{AtomReference, Bridge, Error};
use std::fmt::Write;

/// A solved endpoint experiment. Source and target use the same atom support.
#[derive(Debug, Clone)]
pub struct Study {
    pub source: Vec<f64>,
    pub target: Vec<f64>,
    pub bridge: Bridge,
    pub reference_invariant: Vec<f64>,
    pub reference_transition: Vec<Vec<f64>>,
    pub max_iterations: usize,
    /// Endpoint error before fitting, using the reference transition directly.
    pub baseline_residual: f64,
    pub tolerance: f64,
}
impl Study {
    /// Solve caller-owned distributions against an explicit reversible reference.
    pub fn solve(
        reference: &AtomReference,
        source: &[f64],
        target: &[f64],
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self, Error> {
        let bridge = reference.bridge(source, target, tolerance, max_iterations)?;
        let baseline_residual = (0..source.len())
            .map(|j| {
                ((0..source.len())
                    .map(|i| source[i] * reference.transition()[i][j])
                    .sum::<f64>()
                    - target[j])
                    .abs()
            })
            .fold(0.0_f64, f64::max);
        Ok(Self {
            source: source.to_vec(),
            target: target.to_vec(),
            bridge,
            reference_invariant: reference.invariant().to_vec(),
            reference_transition: reference.transition().to_vec(),
            max_iterations,
            baseline_residual,
            tolerance,
        })
    }
    /// Six atoms with distinct endpoint mass, or equal endpoints for equilibrium.
    pub fn demo(equilibrium: bool) -> Result<Self, Error> {
        let reference = AtomReference::langevin(
            &[0.0, -0.7, 0.4, -0.2, 0.6, -0.4],
            &vec![vec![1.0; 6]; 6],
            1.0,
            0.06,
        )?;
        let source = [0.35, 0.25, 0.15, 0.10, 0.08, 0.07];
        let target = if equilibrium {
            source
        } else {
            [0.06, 0.09, 0.12, 0.18, 0.25, 0.30]
        };
        Self::solve(&reference, &source, &target, 1e-12, 10000)
    }
    pub fn is_equilibrium(&self) -> bool {
        self.source == self.target
    }
    /// Column sums of the solved coupling, to compare with requested target mass.
    pub fn observed_target(&self) -> Vec<f64> {
        (0..self.target.len())
            .map(|j| self.bridge.coupling.iter().map(|row| row[j]).sum())
            .collect()
    }
    pub fn symmetry_residual(&self) -> f64 {
        (0..self.source.len())
            .flat_map(|i| {
                (0..self.source.len())
                    .map(move |j| (self.bridge.coupling[i][j] - self.bridge.coupling[j][i]).abs())
            })
            .fold(0.0_f64, f64::max)
    }
    /// Dependency-free versioned JSON for embedding or independent analysis.
    pub fn json(&self) -> String {
        format!(
            r#"{{"schema":"pudding.study.v1","evidence_kind":"numerical","formal_certificate":false,"crate_version":"{}","reference_invariant":{:?},"reference_transition":{:?},"max_iterations":{},"mode":"{}","source":{:?},"target":{:?},"observed_target":{:?},"coupling":{:?},"transition":{:?},"iterations":{},"tolerance":{},"marginal_residual":{},"baseline_residual":{},"symmetry_residual":{}}}"#,
            env!("CARGO_PKG_VERSION"),
            self.reference_invariant,
            self.reference_transition,
            self.max_iterations,
            if self.is_equilibrium() {
                "equilibrium"
            } else {
                "transport"
            },
            self.source,
            self.target,
            self.observed_target(),
            self.bridge.coupling,
            self.bridge.transition,
            self.bridge.iterations,
            self.tolerance,
            self.bridge.marginal_residual,
            self.baseline_residual,
            self.symmetry_residual()
        )
    }
    /// Standalone SVG. Edge widths encode solved coupling mass, including self edges.
    pub fn svg(&self) -> String {
        let n = self.source.len();
        let height = 180 + n * 64;
        let mut svg = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc"><title id="title">Pudding finite-support bridge</title><desc id="desc">Source atoms on the left, target atoms on the right. Edge widths encode numerical coupling mass. This is not a formal certificate.</desc><rect width="960" height="{height}" rx="18" fill="#101820"/><g fill="#e5eef5" font-family="system-ui,sans-serif"><text x="40" y="40" font-size="24">PUDDING / FINITE-SUPPORT BRIDGE</text><text x="40" y="70" font-size="14">Numerical evidence · endpoint coupling · no formal run certificate</text><text x="60" y="111">SOURCE / requested</text><text x="690" y="111">TARGET / observed · requested</text>"##
        );
        for i in 0..n {
            for j in 0..n {
                let mass = self.bridge.coupling[i][j];
                let y1 = 145 + 64 * i;
                let y2 = 145 + 64 * j;
                write!(svg, r##"<path class="edge" data-mass="{mass}" data-source="{i}" data-target="{j}" d="M 240 {y1} C 420 {y1}, 540 {y2}, 720 {y2}" fill="none" stroke="#69ddbe" stroke-opacity="0.55" stroke-width="{}"><title>atom {i} → {j}: mass {mass:.8}</title></path>"##, 1.0+mass*100.0).unwrap();
            }
        }
        let observed = self.observed_target();
        for (i, observed_mass) in observed.iter().enumerate() {
            let y = 145 + 64 * i;
            write!(svg, r##"<circle cx="240" cy="{y}" r="10" fill="#e9b46d"/><circle cx="720" cy="{y}" r="10" fill="#69ddbe"/><text x="60" y="{}">atom {i} / {:.3}</text><text x="745" y="{}">{:.3} / {:.3}</text>"##,y+5,self.source[i],y+5,observed_mass,self.target[i]).unwrap();
        }
        write!(svg, r#"<text x="40" y="{}" font-size="14">max marginal error {:.2e} · {} scaling iterations</text></g></svg>"#,height-25,self.bridge.marginal_residual,self.bridge.iterations).unwrap();
        svg
    }
    /// Self-contained browser explorer: no server, CDN, telemetry or network calls.
    pub fn html(&self) -> String {
        let template = include_str!("research.html");
        template
            .replace("__STUDY_JSON__", &self.json())
            .replace("__STUDY_SVG__", &self.svg())
    }
}
