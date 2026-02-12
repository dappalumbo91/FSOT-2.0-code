# Day-4 MC — Clock/Stability + Chemistry/Thermal MC
# Chains Day-3 MC averages into Day-4 governors/clocks + RNA→DNA folding statistics

using Random, Statistics, Printf
include("day4.jl")

function compute_S_day4_mc(prev_S::BigFloat, prev_D_eff::BigFloat; delta_psi_base::Float64=-1.8, suction_mult::Float64=3.5)
    # tuned governor bias: stronger yin base and larger suction to favor regulated cycles
    Δψ = BigFloat(string(delta_psi_base)) + BigFloat(randn()*0.30)
    phase_variance = BigFloat("0.18")
    η_eff = φ * (1 + phase_variance)
    recent_hits = BigFloat(25)  # Cycle accumulation

    poof_term = POOF_FACTOR * cos((ψ_con + Δψ)/η_eff) * exp(min(abs(Δψ)/3, BigFloat(50)))
    suction_term = -SUCTION_FACTOR * sin(BigFloat(pi/3.5)) * BigFloat(suction_mult)
    # reduce raw growth so that yin can dominate more often
    growth_term = exp(min(prev_S / BigFloat(10), BigFloat(40)))
    # reduce quirk term so clock coherence doesn't always force positive S
    quirk_term = BigFloat("0.12") * recent_hits * ψ_con

    S = poof_term + suction_term + growth_term + quirk_term
    S = clamp(S, BigFloat(-6), BigFloat(2))  # tighter clamp to favor negative S
    return S
end

function clock_proxy(S::BigFloat)
    governors = S < BigFloat(0)
    regularity = governors ? exp(-Float64(S)) * 99.999 : 1.0
    stability = regularity > 99.9 ? "Pulsar-grade" : "Quasar-variable"
    return (governors=governors, regularity=regularity, stability=stability)
end

function run_day4_mc(; trials::Int=10000, rng_seed::Int=42, base_suction::Float64=3.5, suction_jitter::Float64=0.0, day3_path::String="day3_mc_results.csv")
    Random.seed!(rng_seed)
    # expose jitter to inner loop
    global SUCTION_JITTER = suction_jitter
    global BASE_SUCTION = base_suction
    # use the simple loader (returns Dict with mean_S, mean_gens, timescale_ga, success_frac)
    d3 = load_day3_mc_simple(day3_path)

    gov_count = 0
    fold_count = 0
    regularities = Float64[]
    complexities = Float64[]
    chem_yields = Float64[]
    convs = Float64[]
    eccs = Float64[]
    habitables = 0

    for i in 1:trials
        # apply small per-trial suction jitter to model natural variance and avoid sharp thresholding
        suction_jitter = isdefined(Main, :SUCTION_JITTER) ? Main.SUCTION_JITTER : 0.0
        base_suction = isdefined(Main, :BASE_SUCTION) ? Main.BASE_SUCTION : 3.5
        trial_suction = base_suction * (1.0 + (suction_jitter > 0 ? randn()*suction_jitter : 0.0))
        S = compute_S_day4_mc(BigFloat(d3["mean_S"]), BigFloat(5); suction_mult=trial_suction)
        cp = clock_proxy(S)
        chem = chemical_proxy(BigFloat(d3["mean_gens"]) * BigFloat(1e-56))
        conv = thermal_convection(S, Float64(cp.regularity))
        dna = rna_to_dna(S, Float64(conv), Float64(d3["mean_gens"]))

        # inline orbital proxy to avoid method-age issues
        r_reg = Float64(cp.regularity)
        c_conv = Float64(conv)
        ecc = clamp(1.0 - r_reg/100.0, 0.0, 1.0)
        habitable = (ecc < 0.05) && (c_conv > 0.5)

        if cp.governors
            gov_count += 1
            push!(regularities, Float64(cp.regularity))
        end
        if dna.folding
            fold_count += 1
            push!(complexities, Float64(dna.complexity))
        end
        push!(chem_yields, chem)
        push!(convs, conv)
        push!(eccs, ecc)
        habitables += habitable ? 1 : 0
    end

    gov_rate = gov_count / trials * 100
    fold_rate = fold_count / trials * 100
    mean_reg = isempty(regularities) ? 0.0 : mean(regularities)
    mean_complex = isempty(complexities) ? 0.0 : mean(complexities)
    mean_chem = mean(chem_yields)
    mean_conv = mean(convs)
    mean_ecc = mean(eccs)
    habitable_frac = habitables / trials * 100

    # write CSV summary (one-line aggregate)
    open("day4_mc_summary.csv", "w") do io
        println(io, "trials,gov_rate,fold_rate,mean_reg,mean_complex,mean_chem,mean_conv,mean_ecc,habitable_frac")
        println(io, "$(trials),$(gov_rate),$(fold_rate),$(mean_reg),$(mean_complex),$(mean_chem),$(mean_conv),$(mean_ecc),$(habitable_frac)")
    end

    open("day4_mc_log.txt", "w") do io
        @printf(io, "Day-4 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Governors Active: %.3f%%\n", gov_rate)
        @printf(io, "RNA→DNA folding success: %.3f%%\n", fold_rate)
        @printf(io, "Mean Cycle Regularity: %.12g\n", mean_reg)
        @printf(io, "Mean DNA Complexity proxy: %.6g\n", mean_complex)
        @printf(io, "Mean chemical C/N/O frac: %.6g\n", mean_chem)
        @printf(io, "Mean convection efficiency: %.6g\n", mean_conv)
        @printf(io, "Mean orbital eccentricity (proxy): %.6g\n", mean_ecc)
        @printf(io, "Habitable fraction (proxy): %.3f%%\n", habitable_frac)
    end

    @printf("Day-4 MC (%d trials) — Governors: %.3f%%, Folding: %.3f%%, Mean reg: %.12g\n", trials, gov_rate, fold_rate, mean_reg)

    return Dict("trials"=>trials, "gov_rate"=>gov_rate, "fold_rate"=>fold_rate, "mean_reg"=>mean_reg, "mean_complex"=>mean_complex, "mean_chem"=>mean_chem, "mean_conv"=>mean_conv, "habitable_frac"=>habitable_frac)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day4_mc(trials=10000, rng_seed=42)
end
