# day2_mc.jl — Black-hole valve Monte Carlo for Day‑2 (separation / puncture / FRB proxies)
# Fast MC implementation using Float64 approximations for speed; reproduces refined BH-valve behavior.

using Random, Statistics, Printf

# Reuse Day-1/Day-2 constants by including day1/day2 modules (for anchor/state)
include(joinpath(@__DIR__, "..", "day1", "day1.jl"))
include(joinpath(@__DIR__, "..", "day2.jl"))

# Convert core constants to Float64 for fast MC
const phi_f = Float64(φ)
const psi_con_f = Float64(ψ_con)
const POOF_FACTOR_f = Float64(POOF_FACTOR)
const SUCTION_FACTOR_f = Float64(SUCTION_FACTOR)

# Fast approximated compute_S_day2_bh (Float64)
function compute_S_day2_bh_f(prev_D_eff_f::Float64; delta_psi_base_f::Float64 = -2.618, σ_Δψ::Float64 = 0.4)
    Δψ = delta_psi_base_f + randn() * σ_Δψ
    phase_variance = 0.08 + randn() * 0.05
    η_eff = phi_f * (1 + phase_variance)
    θ_s = pi / 2.0

    # Terms (Float64 approximations of the refined BH-valve model)
    poof_term = POOF_FACTOR_f * cos((psi_con_f + Δψ) / η_eff)
    valve_amplitude = exp(1.0) * (1.0 + 1.0)    # recent_hits ~1 → simple accumulation
    suction_term = -SUCTION_FACTOR_f * (sin(θ_s)^2) * valve_amplitude * 2.5
    growth_term = exp(1.0) * log(prev_D_eff_f + 1.0)

    S = poof_term + suction_term + growth_term

    # puncture/outgas proxies
    puncture = (abs(S) > 4.0) && (S < 0.0)
    outgas = puncture ? exp(-S * 0.8) : 0.0
    frb_proxy = puncture ? (abs(Δψ)^4) * 1e3 : 0.0
    separation = S < 0.0

    return (S=S, separation=separation, puncture=puncture, outgas=outgas, frb_proxy=frb_proxy)
end

# Refined anchor for scale growth (Float64)
function anchor_to_expansion_refined_f(S_f::Float64)
    separation = S_f < 0.0
    if separation
        scale_growth = exp(-S_f * 1.2) * 60.0
    else
        scale_growth = 1.0
    end
    return scale_growth
end

function run_day2_mc(; trials::Int = 10000, seed::Int = 42, delta_psi_base_f::Float64 = -2.618)
    Random.seed!(seed)
    # load previous D_eff from day1_state.json (use numeric fallback)
    day1 = load_day1_state()
    prev_D_eff = haskey(day1, "post_D_eff") ? parse(Float64, day1["post_D_eff"]) : 15.0

    separations = 0
    punctures = 0
    outgasses = Float64[]
    frb_energies = Float64[]
    scale_growths = Float64[]

    for i in 1:trials
        res = compute_S_day2_bh_f(prev_D_eff; delta_psi_base_f=delta_psi_base_f)
        if res.separation
            separations += 1
        end
        if res.puncture
            punctures += 1
            push!(outgasses, res.outgas)
            push!(frb_energies, res.frb_proxy)
        end
        push!(scale_growths, anchor_to_expansion_refined_f(res.S))
    end

    sep_rate = separations / trials * 100.0
    punc_rate = punctures / trials * 100.0
    mean_outgas = length(outgasses) > 0 ? mean(outgasses) : 0.0
    mean_frb = length(frb_energies) > 0 ? mean(frb_energies) : 0.0
    mean_scale = mean(scale_growths)

    @printf("Day 2 MC results (trials=%d, seed=%d)\n", trials, seed)
    @printf("  Separation (S<0) rate: %.2f%%\n", sep_rate)
    @printf("  Puncture (violent BH valve) rate: %.2f%%\n", punc_rate)
    @printf("  Mean outgas (nebula proxy): %.4g\n", mean_outgas)
    @printf("  Mean FRB proxy energy: %.4g\n", mean_frb)
    @printf("  Mean scale_growth (e-fold proxy): %.2f\n", mean_scale)

    # write log
    open("day2_mc_log.txt", "w") do io
        println(io, "Day 2 MC log")
        println(io, "trials = $trials, seed = $seed")
        println(io, "sep_rate = $sep_rate")
        println(io, "punc_rate = $punc_rate")
        println(io, "mean_outgas = $mean_outgas")
        println(io, "mean_frb = $mean_frb")
        println(io, "mean_scale = $mean_scale")
    end

    return Dict("trials"=>trials, "sep_rate"=>sep_rate, "punc_rate"=>punc_rate,
                "mean_outgas"=>mean_outgas, "mean_frb"=>mean_frb, "mean_scale"=>mean_scale)
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    run_day2_mc(trials=10000, seed=42)
end
