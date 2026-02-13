# Day 7 — Post‑Human Rest / Equilibrium (S ≈ 0)
# Models a regulated coherence state after the Humanity image: low strife, sustained query/response cycles

using Printf
using Random, Statistics, CSV, DataFrames
include("day6.jl")   # reuse helper proxies like humanity_proxy if needed

function compute_S_day7(prev_dominion::Float64, observer_activity::Float64)
    # aim for S ≈ 0 (rest); small fluctuations driven by observer queries (activity)
    # prev_dominion scales stability: higher dominion → stronger regulation
    base_noise = 0.002 * (1.0 - clamp(prev_dominion / 1e22, 0.0, 1.0))
    activity_term = 0.01 * clamp(observer_activity, 0.0, 2.0)
    regulation = 0.618  # phi-efficiency prevents runaway

    S = BigFloat( (activity_term * (1.0 - regulation)) * (1.0 - exp(-prev_dominion/1e21)) ) + BigFloat(randn()*base_noise)
    return clamp(S, BigFloat(-0.05), BigFloat(0.05))
end

function rest_proxy(S::BigFloat, dominion::Float64, curiosity::Float64, community_flag::Bool; community_strength::Float64=1.0)
    # Rest stability probability rises with lower |S|, higher dominion and cooperative community
    s_abs = abs(Float64(S))
    stability_prob = clamp(1.0 - s_abs*20.0 + 0.2*log10(1.0 + dominion), 0.0, 1.0)
    # eternal cycle stability: depends on curiosity being present but bounded
    curiosity_term = clamp(curiosity, 0.0, 2.0)
    # allow community_strength to have partial effect even when Day-6 `community_flag` is not explicitly "Cooperative"
    community_baseline = try parse(Float64, get(ENV, "DAY7_COMMUNITY_BASELINE", "0.5")) catch; 0.5 end
    community_effect = community_strength * (community_flag ? 1.0 : community_baseline)

    # --- tuned curiosity-feedback (hump-shaped) to allow a narrow balanced sweet‑spot ---
    # DAY7_CURIO_FEEDBACK_K controls the narrowness of the hump (default 2.0)
    k = try parse(Float64, get(ENV, "DAY7_CURIO_FEEDBACK_K", "2.0")) catch; 2.0 end
    # peak near curiosity=1.0
    curio_effect = curiosity_term * exp(-k * (curiosity_term - 1.0)^2)

    # additional sharpness control (DAY7_CURIO_SHARPNESS > 1 makes the peak narrower)
    sharp_n = try parse(Float64, get(ENV, "DAY7_CURIO_SHARPNESS", "1.0")) catch; 1.0 end
    curio_sharp = curio_effect^sharp_n

    # phi-conjugate scaling (small stabilizing multiplier around 0.618)
    phi_conj = try parse(Float64, get(ENV, "DAY7_PHI_CONJUGATE", "0.618")) catch; 0.618 end

    # baseline & weighting (configurable via ENV for rapid tuning in sweeps)
    baseline = try parse(Float64, get(ENV, "DAY7_CYCLE_BASELINE", "0.30")) catch; 0.30 end
    s_weight = try parse(Float64, get(ENV, "DAY7_S_WEIGHT", "0.20")) catch; 0.20 end
    curio_weight = try parse(Float64, get(ENV, "DAY7_CURIO_WEIGHT", "0.50")) catch; 0.50 end

    # dominion threshold: require sufficiently large dominion to enable high cycle stability
    dom_threshold = try parse(Float64, get(ENV, "DAY7_DOMINION_THRESHOLD", "1e20")) catch; 1e20 end
    dom_factor = dominion >= dom_threshold ? 1.0 : 0.5

    # sharpened, amplitude-scaled contribution from regulated curiosity + community
    cycle_contrib = curio_weight * community_effect * curio_sharp * phi_conj * dom_factor

    cycle_stability = clamp(baseline + s_weight*(1.0 - s_abs) + cycle_contrib, 0.0, 1.0)

    # observer fidelity proxy scales with dominion and regulated curiosity (use curio_sharp)
    fidelity = 10.0^(clamp(log10(dominion + 1.0)/21.0, 0.0, 2.0)) * (1.0 + 0.1*curio_sharp)

    return (rest_stable = (rand() < stability_prob), rest_prob = stability_prob, cycle_stability = cycle_stability, fidelity = fidelity)
end

function run_day7(prev_day6_path::AbstractString = "day6_mc_results.csv"; trials::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)

    if !isfile(prev_day6_path)
        error("Day-6 results not found: " * prev_day6_path)
    end

    d6 = CSV.File(prev_day6_path) |> DataFrame
    N = nrow(d6)

    rows = Vector{Dict{String,Any}}()
    stable_count = 0
    cycle_high = 0
    fidelity_vals = Float64[]

    for i in 1:trials
        # sample a Day-6 trial row uniformly
        prow = d6[rand(1:N), :]
        dominion = try parse(Float64, string(get(prow, :dominion, 1.0))) catch; 1.0 end
        curiosity = try parse(Float64, string(get(prow, :curiosity, 0.5))) catch; 0.5 end
        community_flag = try occursin("Cooperative", string(get(prow, :community_flag, ""))) catch; false end

        # apply scaling parameters (curiosity_scale, community_scale) if present in ENV
        curiosity_scale = try parse(Float64, get(ENV, "DAY7_CURIOSITY_SCALE", "1.0")) catch; 1.0 end
        community_scale = try parse(Float64, get(ENV, "DAY7_COMMUNITY_SCALE", "1.0")) catch; 1.0 end
        S = compute_S_day7(dominion, curiosity * curiosity_scale)
        rp = rest_proxy(S, dominion, curiosity * curiosity_scale, community_flag; community_strength=community_scale)

        push!(rows, Dict("trial"=>i, "S"=>Float64(S), "rest_stable"=>rp.rest_stable, "rest_prob"=>rp.rest_prob, "cycle_stability"=>rp.cycle_stability, "observer_fidelity"=>rp.fidelity))
        push!(fidelity_vals, rp.fidelity)
        if rp.rest_stable
            stable_count += 1
        end
        if rp.cycle_stability > 0.9
            cycle_high += 1
        end
    end

    stable_rate = stable_count / trials * 100
    cycle_rate = cycle_high / trials * 100
    mean_fidelity = mean(fidelity_vals)

    csvp = "day7_mc_results.csv"
    open(csvp, "w") do io
        println(io, "trial,S,rest_stable,rest_prob,cycle_stability,observer_fidelity")
        for r in rows
            println(io, string(r["trial"], ",", r["S"], ",", r["rest_stable"], ",", r["rest_prob"], ",", r["cycle_stability"], ",", r["observer_fidelity"]))
        end
    end

    open("day7_mc_log.txt", "w") do io
        @printf(io, "Day-7 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Rest-stable rate: %.3f%%\n", stable_rate)
        @printf(io, "High-cycle stability (>0.9): %.3f%%\n", cycle_rate)
        @printf(io, "Mean observer fidelity proxy: %.6g\n", mean_fidelity)
    end

    @printf("Day-7 MC summary — trials=%d\nRest-stable: %.3f%% | Cycle-stable: %.3f%% | Mean fidelity: %.6g\n",
            trials, stable_rate, cycle_rate, mean_fidelity)

    return Dict("trials"=>trials, "rest_rate"=>stable_rate, "cycle_rate"=>cycle_rate, "mean_fidelity"=>mean_fidelity)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day7("day6_mc_results.csv", trials=10000, rng_seed=42)
end
