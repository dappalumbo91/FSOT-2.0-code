# day2_mc_extended.jl — Day‑2 Monte Carlo extended with Quasar & Magnetar modes
# - Builds on day2_mc.jl behavior
# - Adds scaled valve_amplitude for Quasar (macro) and Magnetar (micro) analogs
# - Produces detailed statistics and writes `day2_mc_extended_log.txt`

using Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "day1", "day1.jl"))
include("day2_mc.jl")  # reuses helpers where appropriate

# Parameters for modes
const QUASAR_SCALE = 1e6      # multiplicative valve amplitude factor (arbitrary units)
const MAGNETAR_SCALE = 1e-3   # smaller valve amplitude
const QUASAR_PROB = 0.001     # fraction of trials representing quasar-scale events
const MAGNETAR_PROB = 0.05    # fraction representing magnetar-scale events

# Variant of compute_S_day2_bh_f with scale factor input
function compute_S_day2_bh_mode(prev_D_eff_f::Float64, scale_factor::Float64; delta_psi_base_f::Float64 = -2.618)
    # reuse compute_S_day2_bh_f logic but scale valve amplitude
    Δψ = delta_psi_base_f + randn() * 0.4
    phase_variance = 0.08 + randn() * 0.05
    η_eff = Float64(φ) * (1 + phase_variance)
    θ_s = pi / 2.0

    poof_term = Float64(POOF_FACTOR) * cos((Float64(ψ_con) + Δψ) / η_eff)
    # valve amplitude now multiplied by scale_factor
    valve_amplitude = exp(1.0) * (1.0 + 1.0) * scale_factor
    suction_term = -Float64(SUCTION_FACTOR) * (sin(θ_s)^2) * valve_amplitude * 2.5
    growth_term = exp(1.0) * log(prev_D_eff_f + 1.0)

    S = poof_term + suction_term + growth_term
    puncture = (abs(S) > 4.0) && (S < 0.0)
    outgas = puncture ? exp(-S * 0.8) : 0.0
    frb_proxy = puncture ? (abs(Δψ)^4) * 1e3 * scale_factor : 0.0
    separation = S < 0.0
    lensing_proxy = scale_factor > 1e3 ? abs(S) * scale_factor * 1e-6 : 0.0  # crude curvature proxy for quasars

    return (S=S, separation=separation, puncture=puncture, outgas=outgas, frb_proxy=frb_proxy, lensing_proxy=lensing_proxy)
end

function run_day2_mc_extended(; trials::Int = 10000, seed::Int = 42)
    Random.seed!(seed)
    day1 = load_day1_state()
    prev_D_eff = haskey(day1, "post_D_eff") ? parse(Float64, day1["post_D_eff"]) : 15.0

    stats = Dict(
        "total_trials" => 0,
        "separations" => 0,
        "punctures" => 0,
        "quasar_events" => 0,
        "magnetar_events" => 0,
        "outgasses" => Float64[],
        "frb_energies" => Float64[],
        "lensing_values" => Float64[],
        "scale_growths" => Float64[]
    )

    for i in 1:trials
        stats["total_trials"] += 1
        r = rand()
        if r < QUASAR_PROB
            mode = :quasar
            scale = QUASAR_SCALE
            stats["quasar_events"] += 1
        elseif r < QUASAR_PROB + MAGNETAR_PROB
            mode = :magnetar
            scale = MAGNETAR_SCALE
            stats["magnetar_events"] += 1
        else
            mode = :normal
            scale = 1.0
        end

        res = compute_S_day2_bh_mode(prev_D_eff, scale)
        if res.separation
            stats["separations"] += 1
        end
        if res.puncture
            stats["punctures"] += 1
            push!(stats["outgasses"], res.outgas)
            push!(stats["frb_energies"], res.frb_proxy)
        end
        if res.lensing_proxy > 0
            push!(stats["lensing_values"], res.lensing_proxy)
        end
        push!(stats["scale_growths"], anchor_to_expansion_refined_f(res.S))
    end

    # Summarize
    sep_rate = stats["separations"] / trials * 100
    punc_rate = stats["punctures"] / trials * 100
    quasar_count = stats["quasar_events"]
    magnetar_count = stats["magnetar_events"]
    mean_outgas = length(stats["outgasses"]) > 0 ? mean(stats["outgasses"]) : 0.0
    mean_frb = length(stats["frb_energies"]) > 0 ? mean(stats["frb_energies"]) : 0.0
    mean_lensing = length(stats["lensing_values"]) > 0 ? mean(stats["lensing_values"]) : 0.0
    mean_scale = mean(stats["scale_growths"])

    @printf("Day 2 MC Extended (%d trials)\n", trials)
    @printf("  Separation rate: %.2f%%\n", sep_rate)
    @printf("  Puncture rate: %.2f%%\n", punc_rate)
    @printf("  Quasar-mode events: %d (prob %.4f)\n", quasar_count, QUASAR_PROB)
    @printf("  Magnetar-mode events: %d (prob %.4f)\n", magnetar_count, MAGNETAR_PROB)
    @printf("  Mean outgas (nebula proxy): %.4g\n", mean_outgas)
    @printf("  Mean FRB proxy energy: %.4g\n", mean_frb)
    @printf("  Mean lensing proxy (quasar curvature): %.4g\n", mean_lensing)
    @printf("  Mean scale growth (e-fold proxy): %.2f\n", mean_scale)

    open("day2_mc_extended_log.txt", "w") do io
        println(io, "Day2 MC Extended Log")
        println(io, "trials = $trials, seed = $seed")
        println(io, "sep_rate = $sep_rate")
        println(io, "punc_rate = $punc_rate")
        println(io, "quasar_count = $quasar_count")
        println(io, "magnetar_count = $magnetar_count")
        println(io, "mean_outgas = $mean_outgas")
        println(io, "mean_frb = $mean_frb")
        println(io, "mean_lensing = $mean_lensing")
        println(io, "mean_scale = $mean_scale")
    end

    return Dict("sep_rate"=>sep_rate, "punc_rate"=>punc_rate, "quasar_count"=>quasar_count,
                "magnetar_count"=>magnetar_count, "mean_outgas"=>mean_outgas, "mean_frb"=>mean_frb,
                "mean_lensing"=>mean_lensing, "mean_scale"=>mean_scale)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day2_mc_extended(trials=10000, seed=42)
end
