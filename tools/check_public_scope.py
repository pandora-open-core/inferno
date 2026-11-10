#!/usr/bin/env python3
"""Check source disclosure boundaries and the vendored Candle dependency source."""
import json
import re
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
allowed = set(json.loads((root / 'tools/public_modules.json').read_text()))
actual = {str(p.relative_to(root / 'pandorapurple/src')) for p in (root / 'pandorapurple/src').rglob('*.rs')}
assert actual == allowed, f'Module inventory changed: {actual ^ allowed}'
for path in [*(root / 'src').rglob('*.rs'), *(root / 'pandorapurple/src').rglob('*.rs')]:
    code = path.read_text()
    assert code.startswith('//! '), f'Missing library module caption: {path}'
    assert 'codegen' not in path.parts, f'Removed generated backend: {path}'
    assert path.name not in ('cloud.rs', 'coupling_kernel.rs'), f'Excluded integration: {path}'
    assert not re.search(r'pub\s+mod\s+(?:cloud|codegen|coupling_kernel)\s*;', code), f'Excluded module export: {path}'
    assert not re.search(r'\b(?:struct|enum|impl)\s+SchrodingerBridgeGp\b', code), f'Excluded protected GP: {path}'
    assert not re.search(r'pub\s+mod\s+(?:train|reinforce|engine|api|posterior)\s*;', code), f'Excluded public module: {path}'
    assert not re.search(r'\b(?:struct|enum|impl)\s+(?:BridgeObjective|ObjectiveRuntime)\b', code), f'Excluded objective implementation: {path}'

for path in root.rglob('*'):
    if any(part in ('.git', 'target') for part in path.parts):
        continue
    if path.is_file():
        assert path.suffix not in ('.safetensors', '.ckpt', '.pt', '.pth'), f'Weight artifact: {path}'
        assert not any(token in path.name.lower() for token in ('bridge_objective', 'bridge_tables', 'objective_runtime')), f'Excluded artifact: {path}'
assert not (root / 'tools/parity').exists()
manifest = tomllib.loads((root / 'Cargo.toml').read_text())
curated = tomllib.loads((root / 'pandorapurple/Cargo.toml').read_text())
assert curated['package']['autobins'] is False
assert 'serve' not in manifest['features'] and 'serve' not in curated['features']
assert 'cloud' not in manifest['features'] and 'reqwest' not in manifest['dependencies']
assert not (root / 'docs/CLOUD.md').exists()
for name in ('candle-core', 'candle-nn'):
    assert manifest['workspace']['dependencies'][name] == {'path': f'vendor/candle/{name}', 'version': '=0.11.0'}
    assert manifest['dependencies'][name] == {'workspace': True}
assert 'patch' not in manifest, 'Use inherited path dependencies, not consumer-invisible patches'
lock = tomllib.loads((root / 'Cargo.lock').read_text())
for name in ('candle-core', 'candle-nn'):
    packages = [p for p in lock['package'] if p['name'] == name]
    assert len(packages) == 1 and packages[0]['version'] == '0.11.0', f'Unexpected {name}'
    assert 'source' not in packages[0], f'{name} must resolve to the local vendor'
print(f'Public scope OK: {len(actual)} source modules; Candle core/nn use the local vendor.')
