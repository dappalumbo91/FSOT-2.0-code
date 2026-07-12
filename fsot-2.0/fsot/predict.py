"""Uniform FSOT prediction layer for live observables."""

from __future__ import annotations

import re
from typing import Any

from fsot.adapter import domain_scalar, load_compute

ATOMIC_MASS = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Na": 22.99,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "K": 39.098,
    "Ca": 40.078,
    "Fe": 55.845,
    "Br": 79.904,
    "I": 126.904,
}

DOMAIN_FACTORS: dict[str, float] = {
    "Ecology": 0.0002,
    "Biology": 0.0005,
    "Biochemistry": 0.0005,
    "Chemistry": 0.001,
    "Neuroscience": 0.00035,
    "Psychology": 0.0003,
    "Sociology": 0.0002,
    "Economics": 0.0004,
    "Oceanography": 0.0008,
    "Meteorology": 0.0006,
    "Astronomy": 0.00025,
    "Astrophysics": 0.0003,
    "Planetary_Science": 0.0003,
    "Particle_Astrophysics": 0.0002,
    "High_Energy_Physics": 0.00015,
    "Particle_Physics": 0.0001,
    "Materials_Science": 0.0004,
    "Geophysics": 0.0005,
    "Physical_Chemistry": 0.001,
    "Seismology": 0.0005,
}

PROPERTY_ROUTING: dict[str, tuple[str, float]] = {
    "decimalLatitude": ("Ecology", 0.0002),
    "decimalLongitude": ("Ecology", 0.0002),
    "genome_length_bp": ("Biology", 0.0006),
    "molecular_weight": ("Chemistry", 0.001),
    "author_count": ("Particle_Astrophysics", 0.0002),
    "published_year": ("Particle_Astrophysics", 0.0002),
    "heart_rate_bpm": ("Biochemistry", 0.0005),
    "basic_reproduction_R0": ("Biochemistry", 0.0005),
}


def err_pct(computed: float, measured: float) -> float:
    if measured == 0:
        return abs(computed - measured) * 100.0
    return abs(computed - measured) / abs(measured) * 100.0


def formula_mass(formula: str) -> float | None:
    if not formula:
        return None
    total = 0.0
    for elem, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem not in ATOMIC_MASS:
            return None
        n = int(count) if count else 1
        total += ATOMIC_MASS[elem] * n
    return total if total > 0 else None


def route_property(property_name: str, default_domain: str) -> tuple[str, float]:
    if property_name in PROPERTY_ROUTING:
        return PROPERTY_ROUTING[property_name]
    return default_domain, DOMAIN_FACTORS.get(default_domain, 0.001)


def fsot_scaled(measured: float, domain: str, factor: float | None = None) -> tuple[float, float]:
    s = domain_scalar(domain)
    f = factor if factor is not None else DOMAIN_FACTORS.get(domain, 0.001)
    computed = measured * (1.0 + abs(s) * f)
    return computed, err_pct(computed, measured)


def predict_observable(
    measured: float,
    property_name: str,
    *,
    domain: str,
    formula: str | None = None,
    factor: float | None = None,
) -> tuple[float, float, str]:
    routed_domain, routed_factor = route_property(property_name, domain)
    use_factor = factor if factor is not None else routed_factor

    if property_name in {"molecular_weight", "mol_weight"} and formula:
        computed = formula_mass(formula)
        if computed is not None:
            return computed, err_pct(computed, measured), routed_domain

    computed, error = fsot_scaled(measured, routed_domain, use_factor)
    return computed, error, routed_domain


def predict(
    measured: float,
    *,
    property_name: str = "value",
    domain: str = "Biology",
    formula: str | None = None,
) -> dict[str, Any]:
    """High-level predict API for CLI and library use."""
    load_compute()
    computed, error, fsot_domain = predict_observable(
        measured,
        property_name,
        domain=domain,
        formula=formula,
    )
    return {
        "measured": measured,
        "computed": round(computed, 6),
        "error_pct": round(error, 6),
        "property": property_name,
        "domain": domain,
        "fsot_domain": fsot_domain,
        "fsot_scalar": round(domain_scalar(fsot_domain), 6),
    }