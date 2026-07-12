"""Smoke tests for the FSOT 2.0 monolith."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import_compute():
    from fsot.adapter import load_compute

    mod = load_compute()
    assert len(mod.DOMAINS) == 35
    assert float(mod.domain_scalar("Biology")) != 0.0


def test_predict_api():
    from fsot.predict import predict

    out = predict(72.0, property_name="heart_rate_bpm", domain="Biochemistry")
    assert out["error_pct"] < 1.0
    assert out["computed"] > 0


def test_portable_verify():
    from fsot.verify import run_portable_verify

    report = run_portable_verify(verbose=False)
    assert report["domain_count"] == 35
    assert report["ok"], report.get("issues")


def test_cli_verify_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "fsot", "verify", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_manifest_benchmarks_present():
    manifest = json.loads((ROOT / "fsot" / "bundled_data" / "runtime_manifest.json").read_text(encoding="utf-8"))
    for name in manifest["benchmarks"]:
        assert (ROOT / "fsot" / "bundled_data" / "benchmarks" / name).exists()