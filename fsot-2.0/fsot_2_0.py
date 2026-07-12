# FSOT 2.0 legacy shim — prefer: pip install -e . && fsot verify
"""Backward-compatible entry points wrapping the bundled monolith engine."""

from __future__ import annotations

from fsot.adapter import load_compute

_mod = load_compute()

# Re-export canonical constants for notebooks and legacy scripts
phi = _mod.PHI
e = _mod.E
pi = _mod.PI
sqrt2 = _mod.sqrt(2)
log2 = _mod.ln(2)
gamma_euler = _mod.GAMMA
catalan_G = _mod.G_CAT
alpha = _mod.ALPHA
psi_con = _mod.PSI_CON
eta_eff = _mod.ETA_EFF
beta = _mod.BETA
gamma = _mod.GAMMA_C
omega = _mod.OMEGA
theta_s = _mod.THETA_S
poof_factor = _mod.POOF
acoustic_bleed = _mod.A_BLEED
phase_variance = _mod.P_VAR
coherence_efficiency = _mod.C_EFF
bleed_in_factor = _mod.B_IN
acoustic_inflow = _mod.A_IN
suction_factor = _mod.SUCTION
chaos_factor = _mod.CHAOS
perceived_param_base = _mod.P_BASE
new_perceived_param = _mod.P_NEW
consciousness_factor = _mod.C_FACTOR
k = _mod.K

DOMAIN_PARAMS = {
    name.lower().replace(" ", "_"): {
        "D_eff": cfg.D_eff,
        "recent_hits": cfg.hits,
        "delta_psi": float(cfg.delta_psi),
        "delta_theta": float(cfg.delta_theta or 1),
        "observed": cfg.observed,
    }
    for name, cfg in _mod.DOMAINS.items()
}


def compute_S_D_chaotic(
    N=1,
    P=1,
    D_eff=25,
    recent_hits=0,
    delta_psi=1,
    delta_theta=1,
    rho=1,
    scale=1,
    amplitude=1,
    trend_bias=0,
    observed=False,
):
    si = _mod.ScalarInput(
        N=_mod.mpf(N),
        P=_mod.mpf(P),
        D_eff=_mod.mpf(D_eff),
        delta_psi=_mod.mpf(delta_psi),
        delta_theta=_mod.mpf(delta_theta),
        recent_hits=_mod.mpf(recent_hits),
        rho=_mod.mpf(rho),
        scale=_mod.mpf(scale),
        amplitude=_mod.mpf(amplitude),
        trend_bias=_mod.mpf(trend_bias),
        observed=observed,
    )
    return _mod.compute_scalar(si)


def compute_for_domain(domain_name, **overrides):
    aliases = {
        "quantum": "Quantum_Mechanics",
        "biological": "Biology",
        "astronomical": "Astronomy",
        "cosmological": "Cosmology",
    }
    key = domain_name.lower().replace(" ", "_")
    name = domain_name if domain_name in _mod.DOMAINS else aliases.get(key, domain_name)
    if name not in _mod.DOMAINS:
        raise ValueError(f"Unknown domain: {domain_name}. Use `fsot domains` for the 35-domain list.")
    cfg = _mod.DOMAINS[name]
    params = {
        "D_eff": cfg.D_eff,
        "recent_hits": cfg.hits,
        "delta_psi": float(cfg.delta_psi),
        "delta_theta": float(cfg.delta_theta or 1),
        "observed": cfg.observed,
    }
    params.update(overrides)
    return compute_S_D_chaotic(**params)


if __name__ == "__main__":
    print("FSOT 2.0 legacy shim — use: python -m fsot verify")
    print("Cosmology S:", compute_for_domain("Cosmology"))