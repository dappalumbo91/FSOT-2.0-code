"""Portable FSOT verification — constants, domain scalars, bundled benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fsot.adapter import load_compute

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BENCH = DATA / "benchmarks"
MANIFEST = DATA / "runtime_manifest.json"

GATE_POOLED_MEDIAN_PCT = 0.5
GATE_MAX_SCALAR_PCT = 5.0
GATE_MIN_RECORDS = 5
CONST_TOL = 1e-9


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel_err(a: float, b: float) -> float:
    if b == 0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def check_constants(mod: Any) -> list[str]:
    issues: list[str] = []
    cache_path = DATA / "canonical_constants.json"
    if not cache_path.exists():
        return ["canonical_constants.json missing"]
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    mapping = {
        "alpha": mod.ALPHA,
        "psi_con": mod.PSI_CON,
        "eta_eff": mod.ETA_EFF,
        "beta": mod.BETA,
        "k": mod.K,
        "coherence_efficiency": mod.C_EFF,
        "acoustic_bleed": mod.A_BLEED,
        "chaos_factor": mod.CHAOS,
        "consciousness_factor": mod.C_FACTOR,
    }
    for key, live in mapping.items():
        layer = cache.get("layer1", {}).get(key) or cache.get("layer2", {}).get(key)
        if layer is None:
            continue
        if _rel_err(float(live), float(layer)) > CONST_TOL:
            issues.append(f"constant drift: {key}")
    return issues


def check_domains(mod: Any) -> list[str]:
    issues: list[str] = []
    for name in sorted(mod.DOMAINS):
        try:
            float(mod.domain_scalar(name))
        except Exception as exc:
            issues.append(f"domain_scalar failed for {name}: {exc}")
    return issues


def check_wave1(mod: Any) -> list[str]:
    issues: list[str] = []
    for row in mod.wave1():
        if row.measured is None:
            continue
        err = _rel_err(float(row.computed), float(row.measured)) * 100.0
        if err > 15.0:
            issues.append(f"wave1 outlier {row.name}: {err:.2f}%")
    return issues


def _bench_metric(doc: dict, *keys: str, default: float = 99.0) -> float:
    for key in keys:
        if key in doc and doc[key] is not None:
            return float(doc[key])
    return default


def check_benchmark(path: Path) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    doc = json.loads(path.read_text(encoding="utf-8"))
    records = int(doc.get("record_count") or 0)
    pooled = _bench_metric(
        doc,
        "pooled_median_error_pct",
        "headline_median_error_pct",
        "median_error_pct",
    )
    max_scalar = 0.0
    for rec in doc.get("material_records") or []:
        max_scalar = max(max_scalar, float(rec.get("error_pct") or 0))
    if records < GATE_MIN_RECORDS:
        issues.append(f"{path.name}: record_count {records} < {GATE_MIN_RECORDS}")
    if pooled > GATE_POOLED_MEDIAN_PCT:
        issues.append(f"{path.name}: pooled median {pooled:.4f}% > {GATE_POOLED_MEDIAN_PCT}%")
    if max_scalar > GATE_MAX_SCALAR_PCT:
        issues.append(f"{path.name}: max scalar {max_scalar:.4f}% > {GATE_MAX_SCALAR_PCT}%")
    summary = {
        "file": path.name,
        "domain": doc.get("domain"),
        "records": records,
        "pooled_median_error_pct": pooled,
        "max_scalar_error_pct": max_scalar,
        "ok": not issues,
    }
    return summary, issues


def run_portable_verify(*, verbose: bool = True) -> dict[str, Any]:
    mod = load_compute()
    report: dict[str, Any] = {
        "version": "2.0.1",
        "engine": "fsot.compute",
        "domain_count": len(mod.DOMAINS),
        "extension_panel_count": len(getattr(mod, "EXTENSION_FRACTAL_BRANCHES", {})),
        "checks": [],
        "benchmarks": [],
        "ok": True,
    }
    issues: list[str] = []

    for label, fn in (
        ("constants", lambda: check_constants(mod)),
        ("domains", lambda: check_domains(mod)),
        ("wave1", lambda: check_wave1(mod)),
    ):
        chk_issues = fn()
        report["checks"].append({"name": label, "ok": not chk_issues, "issues": chk_issues})
        issues.extend(chk_issues)

    manifest = _load_manifest(MANIFEST)
    bench_files = manifest.get("benchmarks") or sorted(p.name for p in BENCH.glob("*.json"))
    for name in bench_files:
        path = BENCH / name
        if not path.exists():
            issues.append(f"benchmark missing: {name}")
            continue
        summary, bench_issues = check_benchmark(path)
        report["benchmarks"].append(summary)
        issues.extend(bench_issues)
        if verbose:
            tag = "OK" if summary["ok"] else "FAIL"
            print(
                f"  {tag} {summary['domain']}: "
                f"{summary['records']} records, pooled={summary['pooled_median_error_pct']:.4f}%"
            )

    try:
        val = mod.validation_suite()
        val_issues = []
        for row in val:
            if row.measured is None or row.sigma is None:
                continue
            err = _rel_err(float(row.computed), float(row.measured)) * 100.0
            if err > float(row.sigma) * 100 * 2:
                val_issues.append(f"validation {row.name}: {err:.2f}%")
        report["checks"].append({"name": "validation_suite", "ok": not val_issues, "issues": val_issues})
        issues.extend(val_issues)
        if verbose:
            print(f"\n  validation_suite: {len(val)} checks, {len(val_issues)} issues")
    except Exception as exc:
        issues.append(f"validation_suite error: {exc}")

    report["ok"] = not issues
    report["issue_count"] = len(issues)
    report["issues"] = issues
    if verbose:
        print(f"\n=== Portable verify: {'PASS' if report['ok'] else 'FAIL'} ({len(issues)} issues) ===")
        print(f"  35-domain engine: {report['domain_count']} domains")
        print(f"  extension panels: {report['extension_panel_count']}")
    return report