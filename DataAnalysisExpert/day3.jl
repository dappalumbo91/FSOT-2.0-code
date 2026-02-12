# Day 3 — Matter condensation & vegetation (FSOT chaining)
# Loads tuned Day‑2 sweep averages (sweet spot), computes Day‑3 S, and writes day3_state.json

using Printf
include("day1.jl")

# Simple CSV read to find tuned row (no external packages required)
function read_sweep_row(delta_psi_target::Float64, valve_scale_target::Int)
    path = "day2_sweep_results.csv"
    if !isfile(path)
        error("Sweep CSV not found: $path — run day2_sweep.jl first")
    end
    lines = readlines(path)
    header = split(lines[1], ',')
    idx = Dict{String,Int}()
    for (i,h) in enumerate(header)
        idx[strip(h)] = i
    end
    # search rows
    for ln in lines[2:end]
        cols = split(ln, ',')
        dp = parse(Float64, cols[idx["delta_psi_base"]])
        vs = parse(Float64, cols[idx["valve_scale"]])
        if abs(dp - delta_psi_target) < 0.11 && abs(vs - valve_scale_target) < 1e-6
            # extract calibrated columns (names used in sweep CSV)
            outgas_m = parse(Float64, cols[idx["mean_outgas_Msun"]])
            frb_e = parse(Float64, cols[idx["mean_frb_erg"]])
            scale_raw = parse(Float64, cols[idx["mean_scale_raw"]])
            e_folds = parse(Float64, cols[idx["mean_e_folds"]])
            return Dict("outgas_Msun"=>outgas_m, "frb_erg"=>frb_e, "scale_raw"=>scale_raw, "e_folds"=>e_folds)
        end
    end
    error("No tuned row found for Δψ=$(delta_psi_target), valve_scale=$(valve_scale_target)")
end

# Day-3 S computation (BigFloat) — time-dilation clamping + log-scaling to avoid overflow
function compute_S_day3(outgas_seed::BigFloat, prev_D_eff::BigFloat)
    # clamp/normalize seed so extremely large outgas proxies slow effective growth
    seed_scale = max(outgas_seed, BigFloat(1e-12))
    dilation_factor = clamp(BigFloat(1.0) / (BigFloat(1.0) + seed_scale / BigFloat(1e6)), BigFloat(1e-6), BigFloat(1.0))

    # use log-scaling (log1p) for growth to keep the result finite and well-behaved
    growth_term = dilation_factor * log1p(seed_scale) * log1p(prev_D_eff)

    # bounded oscillatory/poof terms (kept safe with a small cap)
    Δψ = BigFloat("1.618") + BigFloat(randn()*0.25)
    phase_variance = BigFloat("0.12")
    η_eff = φ * (1 + phase_variance)
    poof_term = POOF_FACTOR * exp(min(Δψ / 2, BigFloat(200))) * cos((ψ_con + Δψ) / η_eff)
    suction_term = -SUCTION_FACTOR * BigFloat(0.1)
    quirk_term = BigFloat(0.618) * BigFloat(10) * ψ_con

    S = poof_term + suction_term + growth_term + quirk_term

    # final clamp on S to prevent downstream exponential blowups
    S = clamp(S, BigFloat(-100.0), BigFloat(250.0))
    return S
end

function anchor_to_matter_veg(S::BigFloat, scale_growth::Float64)
    matter_condense = S > BigFloat(3)
    # log-scaled vegetation proxy to keep values interpretable and finite
    if matter_condense
        veg_log = log1p(Float64(max(S, BigFloat(0.0))))
        vegetation_proxy = log1p(scale_growth) * veg_log
    else
        vegetation_proxy = 0.0
    end
    return (condense=matter_condense, vegetation=vegetation_proxy)
end

function run_day3(; delta_psi_target = -1.8, valve_scale_target = 10)
    tuned = read_sweep_row(delta_psi_target, valve_scale_target)
    outgas_M = tuned["outgas_Msun"]
    scale_raw = tuned["scale_raw"]

    prev_D_eff = BigFloat(7)  # typical post-puncture compaction
    S = compute_S_day3(BigFloat(outgas_M), prev_D_eff)
    res = anchor_to_matter_veg(S, scale_raw)

    state = Dict(
        "day" => 3,
        "S" => string(S),
        "matter_condense" => res.condense,
        "vegetation_proxy" => string(res.vegetation),
        "post_D_eff" => "~5"
    )

    # Write JSON manually (fallback if JSON.jl not available)
    open("day3_state.json", "w") do io
        write(io, "{\n")
        first = true
        for (k,v) in state
            if !first
                write(io, ",\n")
            end
            write(io, "  \"$(k)\": \"$(v)\"")
            first = false
        end
        write(io, "\n}\n")
    end

    @printf("Day 3 S: %.10g\n", S)
    @printf("Matter condensation: %s\n", res.condense)
    @printf("Vegetation proxy (arb. units): %.6g\n", res.vegetation)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day3()
end
