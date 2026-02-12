# Day-6 MC — Land Animals, Humanity Image & Consciousness stability MC
# Chains Day-5 MC averages into Day-6 outcomes; tracks humanity-image rates + dominion proxy

using Random, Statistics, Printf
include("day6.jl")

function compute_S_day6_mc(diversity::BigFloat, repro_coherence::Float64)
    Δψ = BigFloat("3.618") + BigFloat(randn()*0.6)
    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)

    growth_term = exp(min(diversity / BigFloat(1e9), BigFloat(50))) * BigFloat(0.1)
    poof_term = POOF_FACTOR * exp(min(Δψ, BigFloat(20))) * cos((ψ_con + Δψ)/η_eff)
    suction_term = -SUCTION_FACTOR * BigFloat(0.01)
    quirk_term = (ψ_con * BigFloat(10)) * exp(min(Δψ, BigFloat(10))) * BigFloat(50)

    S = poof_term + suction_term + growth_term + quirk_term
    return clamp(S, BigFloat(-100.0), BigFloat(1e6))
end

function run_day6_mc(; trials::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)
    # load tuned Day-5 aggregates
    d5 = load_day5_mc()
    diversity = d5.diversity
    repro = Float64(d5.repro_coherence)

    rows = Vector{Dict{String,Any}}()
    human_count = 0
    land_count = 0
    dom_vals = Float64[]
    Svals = Float64[]

    for i in 1:trials
        # small jitter in diversity to model ecological variance
        div_j = diversity * (0.8 + 0.4*rand())
        S = compute_S_day6_mc(BigFloat(div_j), repro)
        res = humanity_proxy(S, BigFloat(repro))
        push!(rows, Dict("trial"=>i, "S"=>Float64(S), "land"=>res.animals, "humanity"=>res.humanity, "dominion"=>Float64(res.dominion), "fidelity"=>res.fidelity))
        push!(Svals, Float64(S))
        if res.humanity
            human_count += 1
            push!(dom_vals, Float64(res.dominion))
        end
        if res.animals
            land_count += 1
        end
    end

    human_rate = human_count / trials * 100
    land_rate = land_count / trials * 100
    mean_dom = isempty(dom_vals) ? 0.0 : mean(dom_vals)

    # write CSV
    csvp = "day6_mc_results.csv"
    open(csvp, "w") do io
        println(io, "trial,S,land,humanity,dominion,fidelity")
        for r in rows
            println(io, "$(r["trial"]),$(r["S"]),$(r["land"]),$(r["humanity"]),$(r["dominion"]),\"$(r["fidelity"])\"")
        end
    end

    # write log
    open("day6_mc_log.txt", "w") do io
        @printf(io, "Day-6 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Land animals rate: %.3f%%\n", land_rate)
        @printf(io, "Humanity-image rate: %.3f%%\n", human_rate)
        @printf(io, "Mean dominion proxy (human worlds): %.6g\n", mean_dom)
    end

    # summary stdout
    @printf("Day-6 MC summary — trials=%d\nLand animals: %.3f%% | Humanity-image: %.3f%% | Mean dominion: %.6g\n",
            trials, land_rate, human_rate, mean_dom)

    return Dict("trials"=>trials, "land_rate"=>land_rate, "humanity_rate"=>human_rate, "mean_dominion"=>mean_dom)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day6_mc(trials=10000, rng_seed=42)
end
