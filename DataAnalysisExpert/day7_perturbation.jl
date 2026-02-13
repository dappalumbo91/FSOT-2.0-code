# Day-7 perturbation model — rare variance-driven resets (post‑Sabbath dynamics)
# Uses compute_S_day7() and rest_proxy() from day7.jl

include("day7.jl")
using CSV, DataFrames, Random, Statistics, Printf

function perturbation_trial(d6row::NamedTuple; rng::AbstractRNG=Random.GLOBAL_RNG)
    # read inputs
    dominion = try parse(Float64, string(get(d6row, :dominion, 1.0))) catch; 1.0 end
    curiosity = try parse(Float64, string(get(d6row, :curiosity, 0.9))) catch; 0.9 end
    community_flag = occursin("Cooperative", string(get(d6row, :community_flag, "")))

    # base S and rest/cycle outputs
    S = compute_S_day7(dominion, curiosity)
    rp = rest_proxy(S, dominion, curiosity, community_flag)

    # perturbation parameters (ENV configurable)
    sigma = try parse(Float64, get(ENV, "DAY7_PERTURB_SIGMA", "0.25")) catch; 0.25 end
    threshold = try parse(Float64, get(ENV, "DAY7_PERTURB_THRESHOLD", "1.4")) catch; 1.4 end
    damping_scale = try parse(Float64, get(ENV, "DAY7_COMMUNITY_DAMPING", "0.8")) catch; 0.8 end

    # simulate curiosity jitter and check for over-extension
    jitter = randn(rng) * sigma
    curio_jitter = curiosity + jitter
    perturb_trigger = curio_jitter > threshold

    reset_flag = false
    reset_strength = 0.0
    rebound_success = true
    fidelity_post = rp.fidelity

    if perturb_trigger
        # community damping reduces reset probability
        community_effect = community_flag ? 1.0 : try parse(Float64, get(ENV, "DAY7_COMMUNITY_BASELINE", "0.5")) catch; 0.5 end
        reset_prob = clamp(0.5 + 0.5*(curio_jitter - threshold) - damping_scale*community_effect, 0.0, 1.0)
        if rand(rng) < reset_prob
            reset_flag = true
            # reset strength (how big S spike / dominion perturbation is) — 0..1
            reset_strength = clamp(abs(randn(rng)*0.25 + 0.6), 0.0, 1.0)
            # simulate temporary dominion hit proportional to reset_strength
            dominion_after = dominion * (1.0 - 0.25*reset_strength)
            # recompute S & cycle_stability after reset and after rebound
            S_reset = compute_S_day7(dominion_after, curiosity*0.8)  # curiosity damped transiently
            rp_reset = rest_proxy(S_reset, dominion_after, curiosity*0.8, community_flag)
            # rebound probability increases with community and lower reset_strength
            rebound_prob = clamp(1.0 - 0.6*reset_strength + 0.3*community_effect, 0.0, 1.0)
            rebound_success = rand(rng) < rebound_prob
            if rebound_success
                # partial recovery to near original dominion; fidelity slightly boosted
                fidelity_post = rp_reset.fidelity * (1.0 + 0.08*reset_strength)
            else
                # failed rebound: larger dominion degradation and fidelity loss
                fidelity_post = rp_reset.fidelity * (1.0 - 0.2*reset_strength)
            end
        end
    end

    return (reset_flag=reset_flag, reset_strength=reset_strength, pre_cycle_stability=rp.cycle_stability, post_rebound=rebound_success, fidelity_pre=rp.fidelity, fidelity_post=fidelity_post)
end

function run_day7_perturbation(prev_day6_path::AbstractString="day6_mc_results.csv"; trials::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)
    if !isfile(prev_day6_path)
        error("Day-6 results not found: " * prev_day6_path)
    end

    d6 = CSV.File(prev_day6_path) |> DataFrame
    N = nrow(d6)

    rows = Vector{Dict{String,Any}}()
    reset_count = 0
    rebound_failures = 0
    reset_strengths = Float64[]
    fidelity_change = Float64[]

    for i in 1:trials
        prow = NamedTuple(d6[rand(1:N), :])
        res = perturbation_trial(prow)
        push!(rows, Dict("trial"=>i, "reset_flag"=>res.reset_flag, "reset_strength"=>res.reset_strength, "pre_cycle_stability"=>res.pre_cycle_stability, "post_rebound"=>res.post_rebound, "fidelity_pre"=>res.fidelity_pre, "fidelity_post"=>res.fidelity_post))
        if res.reset_flag
            reset_count += 1
            push!(reset_strengths, res.reset_strength)
            if !res.post_rebound
                rebound_failures += 1
            end
            push!(fidelity_change, (res.fidelity_post - res.fidelity_pre)/res.fidelity_pre)
        end
    end

    reset_rate = reset_count / trials * 100
    rebound_success_rate = reset_count == 0 ? 100.0 : (1.0 - rebound_failures / reset_count) * 100
    mean_reset_strength = isempty(reset_strengths) ? 0.0 : mean(reset_strengths)
    mean_fidelity_boost = isempty(fidelity_change) ? 0.0 : mean(fidelity_change) * 100

    out_csv = "day7_perturbation_results.csv"
    CSV.write(out_csv, DataFrame(rows))

    # plotting is done by Python helper to avoid Julia plotting backend issues (CSV is already written)
    # histogram generation moved to `tools/plot_day7_perturbation.py` (called by the CLI wrapper).

    open("day7_perturbation_log.txt", "w") do io
        @printf(io, "Day-7 perturbation — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Reset rate: %.3f%%\n", reset_rate)
        @printf(io, "Rebound success rate (when reset): %.3f%%\n", rebound_success_rate)
        @printf(io, "Mean reset strength: %.4g\n", mean_reset_strength)
        @printf(io, "Mean fidelity change (%%): %.3f\n", mean_fidelity_boost)
    end

    return Dict("trials"=>trials, "reset_rate"=>reset_rate, "rebound_success_rate"=>rebound_success_rate, "mean_reset_strength"=>mean_reset_strength, "mean_fidelity_change_pct"=>mean_fidelity_boost)
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    res = run_day7_perturbation("day6_mc_results.csv", trials=parse(Int, get(ENV, "DAY7_PERT_TRIALS", "10000")), rng_seed=42)
    println(res)
end
