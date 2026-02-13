# Day-3 Monte Carlo — RNA replication timescale calibration
# Chains tuned Day-2 sweet-spot into Day-3, runs N trials, writes CSV/log/optional plot

using Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "day3.jl"))

function rna_proxy_from_S(S::BigFloat, dilation_factor::Float64=0.618)
    # log-scaled replication generations (calibrated mapping)
    if S > BigFloat(3)
        gens = Float64(log1p(exp(Float64(S) / dilation_factor))) * 1e8
        # convert gens -> timescale proxy (Ga) via empirical calibration
        timescale_ga = max(0.1, 10.0^(log10(gens) - 8.7))  # returns ~0.5-2 Ga for typical gens
    else
        gens = 0.0
        timescale_ga = Inf
    end
    return (gens=gens, timescale_ga=timescale_ga)
end

function run_day3_mc(; trials::Int=10000, rng_seed::Int=42, delta_psi_target=-1.8, valve_scale_target=10)
    Random.seed!(rng_seed)
    tuned = read_sweep_row(delta_psi_target, valve_scale_target)
    outgas_M = tuned["outgas_Msun"]
    scale_raw = tuned["scale_raw"]

    results = Vector{Dict{String,Any}}()
    succ_S = Float64[]
    succ_rna_gens = Float64[]
    succ_times = Float64[]

    for i in 1:trials
        # small jitter to represent ensemble variance in seeds
        seed_jitter = outgas_M * (1.0 + randn()*0.05)
        prev_D_eff = BigFloat(7) + BigFloat(randn()*0.5)
        S = compute_S_day3(BigFloat(seed_jitter), prev_D_eff)
        r = rna_proxy_from_S(S)

        # Planetary accretion + dynamo / atmosphere proxies (added layer)
        accretion_success = (Float64(S) > 3.0) && (rand() > 0.1)  # heuristic
        em_field = accretion_success ? clamp(0.2 + 0.6*rand(), 0.0, 1.0) : 0.0
        atmosphere_retained = em_field > 0.4 && rand() > 0.2
        habitability_boost = atmosphere_retained ? 1.8 : 1.0

        success = S > BigFloat(3)

        # --- short-term tectonic/ocean/weather priors (first-principles heuristics) ---
        # determine a local prior (default 0.5 for baseline run; overwritten by run_day3_mc_with_gaia_priors)
        local_prior = @isdefined(prior) ? prior : 0.5
        # base Earth-like priors scaled by the local habitability prior (higher prior -> stronger active geology/oceans)
        tectonic_rate = clamp(0.5 + 4.0*(local_prior - 0.5), 0.01, 10.0)            # cm/yr proxy
        volcanic_index = clamp(0.2 + 0.8*(local_prior - 0.0), 0.0, 1.0)
        ocean_fraction = clamp(0.4 + 0.6*(local_prior), 0.0, 1.0)
        weather_index = clamp(0.3 + 0.7*(local_prior), 0.0, 1.0)

        push!(results, Dict("trial"=>i,
                            "S"=>Float64(S),
                            "success"=>success,
                            "rna_gens"=>r.gens,
                            "timescale_ga"=>r.timescale_ga,
                            "accretion"=>accretion_success,
                            "em_field"=>em_field,
                            "atmos_retained"=>atmosphere_retained,
                            "habit_boost"=>habitability_boost,
                            "tectonic_rate"=>tectonic_rate,
                            "volcanic_index"=>volcanic_index,
                            "ocean_fraction"=>ocean_fraction,
                            "weather_index"=>weather_index))

        if success
            push!(succ_S, Float64(S))
            push!(succ_rna_gens, r.gens)
            if isfinite(r.timescale_ga)
                push!(succ_times, r.timescale_ga)
            end
        end
    end

    # summary
    success_rate = 100 * length(succ_S) / trials
    mean_S_success = isempty(succ_S) ? NaN : mean(succ_S)
    std_S_success = isempty(succ_S) ? NaN : std(succ_S)
    mean_gens = isempty(succ_rna_gens) ? NaN : mean(succ_rna_gens)
    mean_times = isempty(succ_times) ? Inf : mean(succ_times)

    # write CSV (extended with planetary proxies)
    csv_path = "day3_mc_results.csv"
    open(csv_path, "w") do io
        println(io, "trial,S,success,rna_gens,timescale_ga,accretion,em_field,atmos_retained,habit_boost")
        for row in results
            println(io, "$(row["trial"]),$(row["S"]),$(row["success"]),$(row["rna_gens"]),$(row["timescale_ga"]),$(row["accretion"]),$(row["em_field"]),$(row["atmos_retained"]),$(row["habit_boost"])")
        end
    end

    # write log
    log_path = "day3_mc_log.txt"
    open(log_path, "w") do io
        @printf(io, "Day-3 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Tuned Day-2: Δψ=%.2f, valve_scale=%s\n", delta_psi_target, valve_scale_target)
        @printf(io, "Success rate (S>3): %.3f%%\n", success_rate)
        @printf(io, "Mean S (successes): %.3f ± %.3f\n", mean_S_success, std_S_success)
        @printf(io, "Mean RNA gens (successes): %.3g\n", mean_gens)
        @printf(io, "Mean timescale_ga (successes): %.3g\n", mean_times)
    end

    # optional: attempt to plot (requires Plots.jl)
    try
        if get(ENV, "CI_SKIP_PLOTTING", "0") == "1"
            @info "CI_SKIP_PLOTTING=1 — skipping Plots import in day3_mc"
        else
            @eval using Plots
        end
        svals = [r["S"] for r in results]
        png("day3_mc_hist_S.png") do
            histogram(svals, bins=60, title="Day-3 S distribution", xlabel="S", ylabel="count")
        end
        png("day3_mc_hist_timescale.png") do
            times = filter(x->isfinite(x), [r["timescale_ga"] for r in results])
            histogram(times, bins=50, title="RNA timescale (Ga)", xlabel="timescale (Ga)", ylabel="count")
        end
    catch err
        # plotting optional — ignore if Plots.jl not installed
    end

    # print summary to stdout
    @printf("Day-3 MC summary — trials=%d, seed=%d\n", trials, rng_seed)
    @printf("Success rate (S>3): %.3f%%\n", success_rate)
    @printf("Mean S (successes): %.3f ± %.3f\n", mean_S_success, std_S_success)
    @printf("Mean RNA gens (successes): %.3g\n", mean_gens)
    @printf("Mean timescale_ga (successes): %.3g\n", mean_times)

    return Dict("trials"=>trials, "seed"=>rng_seed, "success_rate"=>success_rate,
                "mean_S_success"=>mean_S_success, "std_S_success"=>std_S_success,
                "mean_gens"=>mean_gens, "mean_timescale_ga"=>mean_times)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day3_mc(trials=10000, rng_seed=42)
end
