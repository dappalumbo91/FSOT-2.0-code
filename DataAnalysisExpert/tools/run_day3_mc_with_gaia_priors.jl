println("Running Day‑3 MC with Gaia priors (resampled from planetary_enriched table)...")
include(joinpath(@__DIR__, "..", "day3", "day3_mc.jl"))
using CSV, DataFrames, Random

# input enriched planetary MC (produced earlier)
planet_csv = get(ENV, "PLANET_CSV", joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k_enriched.csv"))
out_csv = get(ENV, "DAY3_OUT", joinpath(@__DIR__, "..", "day3", "data", "day3_mc_results_gaia.csv"))
trials = parse(Int, get(ENV, "SWEEP_TRIALS", "10000"))
rng = MersenneTwister(parse(Int, get(ENV, "RUN_SEED", "20260212")))

if !isfile(planet_csv)
    error("Enriched planetary CSV not found: ", planet_csv)
end
pdf = CSV.File(planet_csv) |> DataFrame
Np = nrow(pdf)

results = Vector{Dict{String,Any}}()
succ_S = Float64[]
succ_rna_gens = Float64[]
succ_times = Float64[]

# use tuned Day-2 parameters from existing sweep (same as baseline run_day3_mc)
include(joinpath(@__DIR__, "..", "day3.jl"))

for i in 1:trials
    # sample a host prior from enriched planet table
    prow = pdf[rand(rng, 1:Np), :]
    prior = try parse(Float64, string(prow.habitability_enriched)) catch; NaN end
    if isnan(prior)
        prior = 0.275
    end

    # modifiers derived from prior score (calibrated to produce ~+15-25% uplift on high-metal hosts)
    accretion_prob_adj = clamp(0.15 * (prior - 0.5), -0.05, 0.30)   # positive => easier accretion
    em_field_bias = clamp(0.5 * (prior - 0.5), -0.2, 0.5)           # added to em_field when accretion succeeds

    # optional dynamo scaling (sensitivity axis): scale em_field_bias / dynamo coupling
    dynamo_scale = try parse(Float64, get(ENV, "DYNAMO_SCALE", "1.0")) catch; 1.0 end
    em_field_bias *= dynamo_scale

    seed_jitter_scale = 1.0 + 0.06 * (prior - 0.5)                 # small increase in effective seed mass
    timescale_scale = 1.0 - 0.20 * (prior - 0.5)                   # higher prior -> shorter expected timescale

    # replicate core of run_day3_mc but with priors applied
    tuned = read_sweep_row(-1.8, 10)
    outgas_M = tuned["outgas_Msun"]
    seed_jitter = outgas_M * (1.0 + randn(rng)*0.05) * seed_jitter_scale
    prev_D_eff = BigFloat(7) + BigFloat(randn(rng)*0.5)
    S = compute_S_day3(BigFloat(seed_jitter), prev_D_eff)
    r = rna_proxy_from_S(S)

    # accretion and em field modified by priors
    base_accretion_prob = 0.9
    # accretion_success is S>3 AND random pass; we lower the pass threshold when prior is higher
    pass_thresh = clamp(0.1 - accretion_prob_adj, 0.0, 0.95)
    accretion_success = (Float64(S) > 3.0) && (rand(rng) > pass_thresh)

    em_field = accretion_success ? clamp(0.2 + 0.6*rand(rng) + em_field_bias, 0.0, 1.0) : 0.0
    atmosphere_retained = em_field > (0.4 - 0.25*prior) && rand(rng) > clamp(0.2 - 0.15*prior, 0.0, 0.9)
    habitability_boost = atmosphere_retained ? (1.0 + prior) : 1.0

    # adjust timescale by prior influence
    ts = isfinite(r.timescale_ga) ? max(0.01, r.timescale_ga * timescale_scale) : Inf

    success = S > BigFloat(3)
    # capture optional planetary priors (tectonic/ocean/weather) when present in the sampled planetary row
    tectonic_rate_row = try parse(Float64, string(get(prow, :tectonic_rate, NaN))) catch; NaN end
    volcanic_index_row = try parse(Float64, string(get(prow, :volcanic_index, NaN))) catch; NaN end
    ocean_fraction_row = try parse(Float64, string(get(prow, :ocean_fraction, NaN))) catch; NaN end
    weather_index_row = try parse(Float64, string(get(prow, :weather_index, NaN))) catch; NaN end

    push!(results, Dict("trial"=>i,
                        "S"=>Float64(S),
                        "success"=>success,
                        "rna_gens"=>r.gens,
                        "timescale_ga"=>ts,
                        "accretion"=>accretion_success,
                        "em_field"=>em_field,
                        "atmos_retained"=>atmosphere_retained,
                        "habit_boost"=>habitability_boost,
                        "prior"=>prior,
                        "tectonic_rate"=>tectonic_rate_row,
                        "volcanic_index"=>volcanic_index_row,
                        "ocean_fraction"=>ocean_fraction_row,
                        "weather_index"=>weather_index_row))

    if success
        push!(succ_S, Float64(S))
        push!(succ_rna_gens, r.gens)
        if isfinite(ts)
            push!(succ_times, ts)
        end
    end
end

# write Day-3 enriched CSV (same schema as baseline day3_mc_results.csv)
open(out_csv, "w") do io
    println(io, "trial,S,success,rna_gens,timescale_ga,accretion,em_field,atmos_retained,habit_boost,prior,tectonic_rate,volcanic_index,ocean_fraction,weather_index")
    for row in results
        println(io, "$(row["trial"]),$(row["S"]),$(row["success"]),$(row["rna_gens"]),$(row["timescale_ga"]),$(row["accretion"]),$(row["em_field"]),$(row["atmos_retained"]),$(row["habit_boost"]),$(row["prior"]),$(row["tectonic_rate"]),$(row["volcanic_index"]),$(row["ocean_fraction"]),$(row["weather_index"])")
    end
end

# print summary comparable to run_day3_mc
success_rate = 100 * length(succ_S) / trials
mean_S_success = isempty(succ_S) ? NaN : mean(succ_S)
mean_gens = isempty(succ_rna_gens) ? NaN : mean(succ_rna_gens)
mean_times = isempty(succ_times) ? Inf : mean(succ_times)
@printf("Day-3 MC (Gaia priors) summary — trials=%d\n", trials)
@printf("Success rate (S>3): %.3f%%\n", success_rate)
@printf("Mean S (successes): %.3f\n", mean_S_success)
@printf("Mean RNA gens (successes): %.3g\n", mean_gens)
@printf("Mean timescale_ga (successes): %.3g\n", mean_times)

println("Wrote Day-3 enriched CSV: ", out_csv)

# run Day-4 MC using the enriched Day-3 CSV
include(joinpath(@__DIR__, "..", "day4_mc.jl"))
println("Running Day-4 MC chained from enriched Day-3 results...")
res4 = run_day4_mc(trials=10000, rng_seed=42, base_suction=3.5, suction_jitter=0.005, day3_path=out_csv)
println("Day-4 MC result:")
println(res4)
println("Done.")
