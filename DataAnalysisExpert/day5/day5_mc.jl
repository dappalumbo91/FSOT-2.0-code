# Day-5 MC — Biota Swarming & Evolutionary Diversity MC (Darwin track)
# Chains tuned Day-4 averages into Day-5 evolutionary outcomes and writes CSV/log

using Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "day1", "day1.jl"))
include(joinpath(@__DIR__, "..", "day5.jl"))

# Compute a Day-5 S proxy and strife factor (simplified, scaffold)
function compute_S_day5_mc(dna_complex::BigFloat, habitability::Float64)
    recent_hits = BigFloat(25)
    Δψ = BigFloat("2.618") + BigFloat(randn()*0.6)
    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)
    strife_factor = abs(sin(Δψ)) * recent_hits * (1 + BigFloat(habitability))

    # growth term scales with DNA complexity, bounded to avoid overflow
    growth_term = exp(min(dna_complex / BigFloat(1e5), BigFloat(20))) * (strife_factor / BigFloat(10))

    # small poof/suction/quirk scaffold (keeps sign behavior)
    poof_term = POOF_FACTOR * cos((ψ_con + Δψ)/η_eff) * exp(min(Δψ/BigFloat(2), BigFloat(10)))
    suction_term = -SUCTION_FACTOR * BigFloat(0.02)
    quirk_term = BigFloat(0.288) * cos(Δψ)

    S = poof_term + suction_term + growth_term + quirk_term
    return clamp(S, BigFloat(-10), BigFloat(100)), Float64(strife_factor)
end

# Biota proxy: single-cell ignition, multicellular branching, reproductive coherence
function biota_proxy(S::BigFloat, strife_factor::Float64, habitability::Float64)
    single_cell = (S > BigFloat(3)) && (habitability > 0.2)
    multicell_branch = single_cell && (strife_factor > 4.0) ? exp(min(Float64(S)/2, 20.0)) : 1.0
    cambrian_burst = multicell_branch > 50 && rand() < 0.85 ? multicell_branch * exp(strife_factor/10) : multicell_branch
    repro_coherence = cambrian_burst > 1e6 ? "Sperm/egg fidelity active" : "Pre-reproductive"
    diversity = cambrian_burst
    return (swarming=single_cell, multicell=multicell_branch, diversity=diversity, coherence=repro_coherence)
end

function run_day5_mc(; trials::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)
    # attempt to read tuned Day-4 summary; fall back to load_day4_tuned()
    tuned = nothing
    if isfile("day4_mc_summary.csv")
        try
            lines = readlines("day4_mc_summary.csv")
            parts = split(lines[2], ',')
            tuned = Dict("gov_rate"=>parse(Float64, parts[2]), "mean_reg"=>parse(Float64, parts[4]), "mean_complex"=>parse(Float64, parts[5]), "habitable_frac"=>parse(Float64, parts[9]))
        catch
            tuned = nothing
        end
    end
    if tuned === nothing
        td = load_day4_tuned()
        tuned = Dict("gov_rate"=>99.0, "mean_reg"=>Float64(td.regularity), "mean_complex"=>Float64(td.dna_complexity), "habitable_frac"=>83.0)
    end

    # per-trial storage
    rows = Vector{Dict{String,Any}}(undef, 0)
    scount = 0
    multicount = 0
    cambrian_count = 0
    fidelity_count = 0
    diversity_vals = Float64[]

    for i in 1:trials
        # sample habitability for this trial from tuned habitable fraction
        base_hab = rand() < tuned["habitable_frac"]/100 ? tuned["mean_reg"]/100.0 : tuned["mean_reg"]/300.0
        # boost from ocean_fraction + weather_index when available (short-term integration)
        ocean_boost = haskey(tuned, "mean_ocean_fraction") && !isnan(tuned["mean_ocean_fraction"]) ? 0.18 * tuned["mean_ocean_fraction"] : 0.0
        weather_boost = haskey(tuned, "mean_weather_index") && !isnan(tuned["mean_weather_index"]) ? 0.12 * tuned["mean_weather_index"] : 0.0
        hab = clamp(base_hab * (1.0 + ocean_boost + weather_boost), 0.0, 1.0)

        dna_complex = tuned["mean_complex"] * (0.5 + rand()*1.0)  # jitter complexity

        S, strife = compute_S_day5_mc(BigFloat(dna_complex), hab)
        b = biota_proxy(S, strife, hab)

        push!(rows, Dict("trial"=>i, "S"=>Float64(S), "strife"=>strife, "hab"=>hab, "swarming"=>b.swarming, "multicell"=>b.multicell, "diversity"=>b.diversity, "coherence"=>b.coherence))

        if b.swarming; scount += 1; end
        if b.multicell > 1.0; multicount += 1; end
        if b.diversity > 1e3; cambrian_count += 1; end
        if b.coherence == "Sperm/egg fidelity active"; fidelity_count += 1; end
        push!(diversity_vals, Float64(b.diversity))
    end

    # summary
    swarming_rate = scount / trials * 100
    multicell_rate = multicount / trials * 100
    cambrian_rate = cambrian_count / trials * 100
    fidelity_rate = fidelity_count / trials * 100
    mean_diversity = mean(diversity_vals)

    # write CSV
    csvp = "day5_mc_results.csv"
    open(csvp, "w") do io
        println(io, "trial,S,strife,hab,swarming,multicell,diversity,coherence")
        for r in rows
            println(io, "$(r["trial"]),$(r["S"]),$(r["strife"]),$(r["hab"]),$(r["swarming"]),$(r["multicell"]),$(r["diversity"]),\"$(r["coherence"])\"")
        end
    end

    # write log
    open("day5_mc_log.txt", "w") do io
        @printf(io, "Day-5 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Swarming biota rate (S>3): %.3f%%\n", swarming_rate)
        @printf(io, "Multicellular branching rate: %.3f%%\n", multicell_rate)
        @printf(io, "Cambrian-like burst rate: %.3f%%\n", cambrian_rate)
        @printf(io, "Reproductive coherence (sperm/egg): %.3f%%\n", fidelity_rate)
        @printf(io, "Mean diversity proxy: %.6g\n", mean_diversity)
    end

    @printf("Day-5 MC summary — trials=%d\nSwarming: %.3f%% | Multicell: %.3f%% | Cambrian: %.3f%% | Fidelity: %.3f%%\nMean diversity: %.6g\n",
            trials, swarming_rate, multicell_rate, cambrian_rate, fidelity_rate, mean_diversity)

    return Dict("trials"=>trials, "swarming_rate"=>swarming_rate, "multicell_rate"=>multicell_rate, "cambrian_rate"=>cambrian_rate, "fidelity_rate"=>fidelity_rate, "mean_diversity"=>mean_diversity)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day5_mc(trials=10000, rng_seed=42)
end
