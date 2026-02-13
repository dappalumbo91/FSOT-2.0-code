# Day 4 — Lights / Governors / Time Standardization (scaffolding)
# Chains Day‑3 MC outputs into a Day‑4 governor + time‑regularity proxy

include("day3.jl")   # provides compute_S_day3 helpers
using Statistics

# Simple CSV loader (no external CSV.jl dependency required)
function load_day3_mc_simple(path::String="day3_mc_results.csv")
    if !isfile(path)
        error("Day‑3 MC CSV not found: $path")
    end
    lines = readlines(path)
    header = split(lines[1], ',')
    idx = Dict{String,Int}()
    for (i,h) in enumerate(header)
        idx[strip(h)] = i
    end
    S_vals = Float64[]
    gens = Float64[]
    times = Float64[]
    successes = 0
    for ln in lines[2:end]
        cols = split(ln, ',')
        s = parse(Float64, cols[idx["S"]])
        succ = cols[idx["success"]] in ("true","True","1")
        push!(S_vals, s)
        if succ
            successes += 1
            push!(gens, parse(Float64, cols[idx["rna_gens"]]))
            t = cols[idx["timescale_ga"]]
            if t != "Inf"
                push!(times, parse(Float64, t))
            end
        end
    end
    mean_S = mean(S_vals)
    mean_gens = isempty(gens) ? 0.0 : mean(gens)
    mean_time_ga = isempty(times) ? Inf : mean(times)

    # attempt to extract optional tectonic/ocean/weather columns if present
    tect_rates = Float64[]
    ocean_fracs = Float64[]
    weather_idxs = Float64[]
    volc_idxs = Float64[]
    for ln in lines[2:end]
        cols = split(ln, ',')
        # guard: if extra columns exist at end, attempt to read by name via header mapping
        if haskey(idx, "tectonic_rate") && cols[idx["tectonic_rate"]] != ""
            push!(tect_rates, parse(Float64, cols[idx["tectonic_rate"]]))
        end
        if haskey(idx, "ocean_fraction") && cols[idx["ocean_fraction"]] != ""
            push!(ocean_fracs, parse(Float64, cols[idx["ocean_fraction"]]))
        end
        if haskey(idx, "weather_index") && cols[idx["weather_index"]] != ""
            push!(weather_idxs, parse(Float64, cols[idx["weather_index"]]))
        end
        if haskey(idx, "volcanic_index") && cols[idx["volcanic_index"]] != ""
            push!(volc_idxs, parse(Float64, cols[idx["volcanic_index"]]))
        end
    end

    return Dict("mean_S"=>mean_S, "mean_gens"=>mean_gens, "timescale_ga"=>mean_time_ga, "success_frac"=>successes/length(S_vals),
                "mean_tectonic_rate"=>isempty(tect_rates) ? NaN : mean(tect_rates),
                "mean_ocean_fraction"=>isempty(ocean_fracs) ? NaN : mean(ocean_fracs),
                "mean_weather_index"=>isempty(weather_idxs) ? NaN : mean(weather_idxs),
                "mean_volcanic_index"=>isempty(volc_idxs) ? NaN : mean(volc_idxs))
end

function compute_S_day4(prev_S::BigFloat, prev_D_eff::BigFloat)
    # mild yin damping for governors; keep terms bounded
    Δψ = BigFloat("-0.618") + BigFloat(randn()*0.3)
    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)

    poof_term = POOF_FACTOR * cos((ψ_con + Δψ)/η_eff) * exp(min(Δψ / 2, BigFloat(50)))
    suction_term = -SUCTION_FACTOR * sin(BigFloat(pi/4)) * BigFloat(1.2)
    growth_term = log1p(prev_S)  # carry small momentum from Day 3
    quirk_term = (isdefined(Main, :observed) && observed) ? ψ_con * BigFloat(20) : BigFloat(0.618) * BigFloat(20)

    S = poof_term + suction_term + growth_term + quirk_term
    S = clamp(S, BigFloat(-5.0), BigFloat(5.0))
    return S
end

function time_proxy(S::BigFloat, timescale_ga::Float64)
    governor_success = S < BigFloat(0)
    cycle_regularity = governor_success ? exp(-Float64(S)) * 100.0 : 1.0
    standardized_time_years = isfinite(timescale_ga) ? timescale_ga * 1e9 / max(cycle_regularity, 1e-6) : Inf
    return (governors=governor_success, regularity=cycle_regularity, time_years=standardized_time_years)
end

# ---- Chemistry / Thermal / RNA→DNA extensions ----
function chemical_proxy(outgas_mass::BigFloat)
    # anchored mapping: heavier-element (C/N/O) proxy from outgas mass
    # uses soft-exponential to avoid runaway and returns fraction [0,1]
    fusion_yield = clamp(Float64(exp(min(outgas_mass / BigFloat(1e65), BigFloat(10)))) * 0.12, 0.0, 1.0)
    return fusion_yield
end

function thermal_convection(S::BigFloat, regularity::Float64)
    # map radiation/poof S -> convection efficiency in [0,1]
    radiation_poof = exp(max(Float64(S), 0.0))
    convection = clamp((radiation_poof * regularity / φ) / 10.0, 0.0, 1.0)
    return convection
end

# --- tectonic / volcanic stochastic processes (puncture / rebound mechanics) ---
function tectonic_event(tectonic_rate::Float64; rng=GLOBAL_RNG)
    # tectonic_rate in cm/yr; returns (puncture_prob, rebound_strength)
    # higher rate -> more frequent punctures and stronger rebound
    base_prob = clamp(tectonic_rate / 10.0, 0.01, 0.9)
    puncture = rand(rng) < base_prob
    rebound = puncture ? clamp(0.5 + 0.5*rand(rng)*(tectonic_rate/5.0), 0.1, 2.0) : 0.0
    return (puncture=puncture, rebound=rebound)
end

function volcanic_outgassing(volcanic_index::Float64, rebound::Float64)
    # volcanic_index [0..1], rebound amplifies transient outgas
    outgas = clamp(volcanic_index * (1.0 + rebound), 0.0, 2.0)
    # convert to chemical yield modifier
    chem_boost = clamp(0.1 + 0.4*outgas, 0.0, 1.0)
    return (outgas=outgas, chem_boost=chem_boost)
end

function rna_to_dna(S::BigFloat, convection::Float64, rna_gens::Float64; Δψ::Float64=-0.618, recent_hits::Int=20)
    # strife/tribulation increases folding chances; require sufficient convection + RNA generations
    strife_factor = abs(sin(Δψ)) * recent_hits
    folding_success = (convection > 0.5) && (rna_gens > 1e8)
    dna_complexity = folding_success ? exp(min(strife_factor, 20.0)) * (rna_gens / 1e8) : 1.0
    return (folding=folding_success, complexity=dna_complexity)
end

function run_day4()
    d3 = load_day3_mc_simple()
    prev_D_eff = BigFloat(5)
    S = compute_S_day4(BigFloat(d3["mean_S"]), prev_D_eff)
    res = time_proxy(S, Float64(d3["timescale_ga"]))

    # chemical / thermal / RNA→DNA chaining
    chem = chemical_proxy(BigFloat(d3["mean_gens"]) * BigFloat(1e-56))  # scale gens -> mass proxy
    conv = thermal_convection(S, Float64(res.regularity))
    rna_res = rna_to_dna(S, Float64(conv), Float64(d3["mean_gens"]))

    state = Dict(
        "day" => 4,
        "S" => string(S),
        "governors" => string(res.governors),
        "cycle_regularity_percent" => string(res.regularity),
        "standardized_time_years" => string(res.time_years),
        "chemical_CN_O_frac" => string(chem),
        "convection_efficiency" => string(conv),
        "rna_to_dna_folding" => string(rna_res.folding),
        "dna_complexity_proxy" => string(rna_res.complexity)
    )

    # safe JSON write without JSON.jl
    open("day4_state.json", "w") do io
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

    @printf("Day 4 S: %.10g\n", S)
    @printf("Governors Active: %s\n", res.governors)
    @printf("Cycle Regularity: %.6g%%\n", res.regularity)
    @printf("Chemical C/N/O fraction proxy: %.6g\n", chem)
    @printf("Convection efficiency: %.6g\n", conv)
    @printf("RNA→DNA folding success: %s\n", rna_res.folding)
    @printf("DNA complexity proxy: %.6g\n", rna_res.complexity)

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day4()
end
