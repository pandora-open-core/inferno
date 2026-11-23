#!/usr/bin/env python3
"""Check source disclosure boundaries and the registry dependency sources."""
import json
import os
import re
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
allowed = set(json.loads((root / 'tools/public_modules.json').read_text()))
actual = {str(p.relative_to(root / 'src/pandora')) for p in (root / 'src/pandora').rglob('*.rs')}
assert actual == allowed, f'Module inventory changed: {actual ^ allowed}'
for path in (root / 'src').rglob('*.rs'):
    code = path.read_text()
    assert code.startswith('//! '), f'Missing library module caption: {path}'
    assert 'codegen' not in path.parts, f'Removed generated backend: {path}'
    assert path.name not in ('cloud.rs', 'coupling_kernel.rs'), f'Excluded integration: {path}'
    assert not re.search(r'pub\s+mod\s+(?:cloud|codegen|coupling_kernel)\s*;', code), f'Excluded module export: {path}'
    assert not re.search(r'\b(?:struct|enum|impl)\s+SchrodingerBridgeGp\b', code), f'Excluded protected GP: {path}'
    checked_code = code.replace('pub mod engine;', '') if path == root / 'src/autograd/mod.rs' else code
    assert not re.search(r'pub\s+mod\s+(?:train|reinforce|engine|api|posterior)\s*;', checked_code), f'Excluded public module: {path}'
    assert not re.search(r'\b(?:struct|enum|impl)\s+(?:BridgeObjective|ObjectiveRuntime)\b', code), f'Excluded objective implementation: {path}'

for base, directories, filenames in os.walk(root):
    directories[:] = [d for d in directories if d not in ('.git', 'target', '.lake', '__pycache__')]
    for filename in filenames:
        path = Path(base) / filename
        assert path.suffix not in ('.safetensors', '.ckpt', '.pt', '.pth'), f'Weight artifact: {path}'
        assert not any(token in path.name.lower() for token in ('bridge_objective', 'bridge_tables', 'objective_runtime')), f'Excluded artifact: {path}'
assert not (root / 'tools/parity').exists()
manifest = tomllib.loads((root / 'Cargo.toml').read_text())
assert 'serve' not in manifest['features'] and 'cloud' not in manifest['features']
assert 'reqwest' not in manifest['dependencies']
assert 'patch' not in manifest and 'workspace' not in manifest
for name, dependency in manifest['dependencies'].items():
    assert not isinstance(dependency, dict) or not any(k in dependency for k in ('path', 'git')), name
lock = tomllib.loads((root / 'Cargo.lock').read_text())
for name in ('candle-core', 'candle-nn'):
    packages = [p for p in lock['package'] if p['name'] == name]
    assert len(packages) == 1 and packages[0]['version'] == '0.11.1'
    assert packages[0]['source'].startswith('registry+')
print(f'Public scope OK: {len(actual)} curated model modules; all dependencies use the registry.')
