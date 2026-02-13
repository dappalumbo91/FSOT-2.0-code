# Per-trial Day-4 diagnostic
# Writes a CSV with per-trial fields so we can inspect failing trials.
using CSV, DataFrames, Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "day4.jl"))
include(joinpath(@__DIR__, "..", "day4_mc.jl"))

day3_path = get(ENV, "DAY3_PATH", joinpath(@__DIR__, "..", "data", "day3_mc_results_gaia_scaled_0p75.csv"))
trials = parse(Int, get(ENV, "DIAG_TRIALS", "50000"))
rng_seed = parse(Int, get(ENV, "DIAG_SEED", "20260212"))

# load day3 trials if present, else fallback to load_day3_mc_simple
if isfile(day3_path)
    df3 = CSV.File(day3_path) |> DataFrame
    mean_S = mean(Float64.(df3.S))
    mean_gens = mean(Float64.(df3.rna_gens))
else
    d3 = load_day3_mc_simple(day3_path)  # uses existing loader
    mean_S = Float64(d3["mean_S"])
    mean_gens = Float64(d3["mean_gens"])
end

println("Day-3 mean_S = ", mean_S)

Random.seed!(rng_seed)
rows = Vector{Dict{String,Any}}()
for i in 1:trials
    # copy logic from run_day4_mc inner loop
    suction_jitter = isdefined(Main, :SUCTION_JITTER) ? Main.SUCTION_JITTER : 0.0
    base_suction = isdefined(Main, :BASE_SUCTION) ? Main.BASE_SUCTION : 3.5
    trial_suction = base_suction * (1.0 + (suction_jitter > 0 ? randn()*suction_jitter : 0.0))
    S = compute_S_day4_mc(BigFloat(mean_S), BigFloat(5); suction_mult=trial_suction)
    cp = clock_proxy(S)
    chem = chemical_proxy(BigFloat(mean_gens) * BigFloat(1e-56))
    conv = thermal_convection(S, Float64(cp.regularity))
    dna = rna_to_dna(S, Float64(conv), Float64(mean_gens))
    r_reg = Float64(cp.regularity)
    c_conv = Float64(conv)
    ecc = clamp(1.0 - r_reg/100.0, 0.0, 1.0)
    habitable = (ecc < 0.05) && (c_conv > 0.5)

    push!(rows, Dict(
        "trial"=>i,
        "S"=>Float64(S),
        "governors"=>cp.governors,
        "regularity"=>r_reg,
        "stability"=>cp.stability,
        "chem_yield"=>chem,
        "convection"=>c_conv,
        "dna_folding"=>dna.folding,
        "dna_complexity"=>Float64(dna.complexity),
        "ecc"=>ecc,
        "habitable"=>habitable
    ))
end

out = joinpath(@__DIR__, "..", "sensitivity", "day4_diagnostic_m0p75_$(trials).csv")
mkpath(dirname(out))
CSV.write(out, DataFrame(rows))
println("Wrote diagnostic CSV: ", out)
