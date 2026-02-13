# Day 2 — FSOT 2.0: Firmament / Separation (Julia)
# Chains from Day 1 state (day1_state.json), computes Day-2 S and anchors to an
# expansion/inflation proxy. Writes day2_state.json.

using Printf
# lightweight JSON fallback like day1.jl
has_json = try
    @eval using JSON; true
catch
    false
end

# Import Day-1 helpers by reading state file; reuse constants from day1.jl where needed
include(joinpath(@__DIR__, "day1", "day1.jl"))

const REAL_H0 = BigFloat("67.4")
const REAL_NS = BigFloat("0.965")
const INFLATION_EFOLDS_PROXY = BigFloat("55")

function load_day1_state(path::String = "day1_state.json")
    if !isfile(path)
        error("Day 1 state not found — run day1/day1.jl first")
    end
    content = read(path, String)
    if has_json
        st = JSON.parse(content)
        return st
    else
        # minimal parse for our known keys
        m = Dict{String,String}()
        for line in split(content, '\n')
            if occursin(":", line)
                parts = split(line, ":")
                k = replace(strip(parts[1]), '"' => "")
                v = strip(join(parts[2:end], ":"))
                v = replace(v, '\"' => "")
                v = replace(v, ',' => "")
                m[string(k)] = strip(v)
            end
        end
        return m
    end
end

function compute_S_day2(prev_D_eff::BigFloat)
    Δψ = BigFloat("-0.618")  # suction bias for separation (yin peak)
    phase_variance = BigFloat("0.05")
    η_eff = φ * (1 + phase_variance)
    θ_s = pi / BigFloat(3)

    poof_term = POOF_FACTOR * cos((ψ_con + Δψ) / η_eff) * exp(Δψ / 2)
    suction_term = -SUCTION_FACTOR * sin(θ_s) * ((prev_D_eff - 10)/15 + 1) * BigFloat(1.5)
    growth_term = exp(BigFloat(1)) * log(prev_D_eff + 1)
    quirk_term = cos(Δψ) * BigFloat(0.618)

    S = poof_term + suction_term + growth_term + quirk_term
    return S
end

function anchor_to_expansion(S::BigFloat, prev_density::BigFloat)
    separation = S < 0  # Suction dominant = division success
    scale_factor_growth = separation ? INFLATION_EFOLDS_PROXY * exp(-S/BigFloat(2)) : BigFloat(1)
    anchored_horizon = scale_factor_growth * REAL_H0
    return (separation=separation, scale_growth=scale_factor_growth, anchored_expansion=anchored_horizon)
end

function run_day2(; save_state::Bool = true)
    day1 = load_day1_state()
    prev_D_eff = haskey(day1, "post_D_eff") ? BigFloat(day1["post_D_eff"]) : BigFloat(15)
    prev_density = haskey(day1, "anchored_density_J_per_m3") ? BigFloat(day1["anchored_density_J_per_m3"]) : REAL_RHO_R_TODAY

    S = compute_S_day2(prev_D_eff)
    res = anchor_to_expansion(S, prev_density)
    post_D_eff = res.separation ? prev_D_eff - BigFloat(8) : prev_D_eff

    state = Dict(
        "day" => 2,
        "S" => string(S),
        "separation" => res.separation,
        "scale_growth_e_folds_proxy" => string(res.scale_growth),
        "anchored_expansion_H0_proxy" => string(res.anchored_expansion),
        "post_D_eff" => string(post_D_eff)
    )

    if save_state
        if has_json
            open("day2_state.json", "w") do io
                write(io, JSON.json(state; indent=4))
            end
        else
            open("day2_state.json", "w") do io
                for (k,v) in state
                    println(io, "$(k): $(v)")
                end
            end
        end
    end

    @printf("Day 2 S: %.10g\n", S)
    @printf("Separation (firmament formed): %s\n", res.separation)
    @printf("Anchored Expansion Proxy (H0 * e-fold proxy): %s\n", string(res.anchored_expansion))
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day2()
end
