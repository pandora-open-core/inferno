#!/usr/bin/env python3
"""Compile and run an external Cargo consumer without patch tables."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="inferno-consumer-") as directory:
    project = Path(directory)
    (project / "src").mkdir()
    (project / "Cargo.toml").write_text(
        '[package]\nname = "inferno-alpha-consumer"\nversion = "0.0.0"\nedition = "2021"\n'
        '[dependencies]\ninferno = { package = "inferno-rs", path = ' + json.dumps(str(root)) + ' }\n'
    )
    (project / "src/main.rs").write_text(r'''use inferno::prelude::*;
fn main() -> Result<()> {
    let d = Device::Cpu;
    let model = Model::new(&d, |vb| nn::linear(1, 1, vb))?;
    let x = Tensor::new(&[[0f32], [1.]], &d)?;
    assert_eq!(model.forward(&x)?.dims(), &[2, 1]);
    let observation = inferno::diagnostics::snapshot(&x)?;
    let mut engine = inferno::pudding::Engine::new();
    engine.check("finite model input", inferno::pudding::Check::Finite(&observation));
    assert!(engine.finish().passed());
    let _: Option<inferno::models::PandoraGP> = None;
    let _: Option<inferno::pandorapurple::gp::DifferentiableGp> = None;
    let _vae_constructor = inferno::models::PandoraVae::build;
    println!("external consumer: Candle model and Pudding diagnostics passed");
    Ok(())
}
''')
    env = dict(os.environ)
    env.setdefault("CARGO_TARGET_DIR", str(root / "target"))
    subprocess.run(["cargo", "run", "--manifest-path", str(project / "Cargo.toml")], env=env, check=True)
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--manifest-path", str(project / "Cargo.toml")], env=env
    ))
    for name in ("candle-core", "candle-nn"):
        packages = [p for p in metadata["packages"] if p["name"] == name]
        assert len(packages) == 1 and packages[0]["source"] is None, packages
        assert Path(packages[0]["manifest_path"]).is_relative_to(root / "vendor/candle")
