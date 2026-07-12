# FSOT 2.0 Monolith

Interpretable, independent FSOT runtime — **Python + mpmath only** for the core engine.

## Quick start

```bash
git clone https://github.com/dappalumbo91/FSOT-2.0-code.git
cd FSOT-2.0-code/fsot-2.0
python -m pip install --upgrade pip
pip install -e .
fsot verify          # portable constants + bundled benchmark replay
fsot domains         # list 35 domain scalars
fsot predict 72 --property heart_rate_bpm --domain Biochemistry
fsot run --validate  # 28-check validation suite from the compute engine
```

For CI-parity checks: `pip install -e ".[dev]"` then `pytest tests/test_monolith.py -q`.

## Commands

| Command | Purpose |
|---------|---------|
| `fsot verify` | Check canonical constants, 35 domains, wave-1 targets, bundled panels |
| `fsot predict` | FSOT prediction for one measured observable |
| `fsot domains` | Print all domain names, D_eff, and scalars |
| `fsot run` | Full FSOT compute engine (waves, chemistry, consciousness, etc.) |

## Layout

```
fsot/
  compute.py   # authoritative scalar engine (from FSOT 2.1)
  predict.py   # live-observable prediction layer
  verify.py    # portable verification (no Lean/Coq required)
  cli.py       # entrypoint
data/
  canonical_constants.json
  runtime_manifest.json
  benchmarks/  # bundled panel replays
```

## Legacy

`fsot_2_0.py` remains as a thin compatibility shim. Document ingestion (`main.py`) and Julia MC tools are unchanged optional extras.

## Requirements

- Python 3.10+
- `mpmath` (only hard dependency for the monolith)