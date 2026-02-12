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
    return Dict("mean_S"=>mean_S, "mean_gens"=>mean_gens, "timescale_ga"=>mean_time_ga, "success_frac"=>successes/length(S_vals))
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

function run_day4()
    d3 = load_day3_mc_simple()
    prev_D_eff = BigFloat(5)
    S = compute_S_day4(BigFloat(d3["mean_S"]), prev_D_eff)
    res = time_proxy(S, Float64(d3["timescale_ga"]))

    state = Dict(
        "day" => 4,
        "S" => string(S),
        "governors" => string(res.governors),
        "cycle_regularity_percent" => string(res.regularity),
        "standardized_time_years" => string(res.time_years)
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
    @printf("Standardized Time Proxy: %.10g years\n", res.time_years)

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day4()
end
