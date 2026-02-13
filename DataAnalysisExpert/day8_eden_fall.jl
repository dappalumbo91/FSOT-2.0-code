# Day 8 — Eden / Fall MC (multi‑trial strife episodes + recovery dynamics)
# Builds on tuned Day‑7 + perturbation tails (day7.jl + day7_perturbation.jl)

include("day7_perturbation.jl")
using CSV, DataFrames, Random, Statistics, Printf

# Strife episode proxy: length (cycles) drawn from a distribution that scales with reset_strength
# Recovery (rebound) probability increases with community and decreases with reset_strength

function simulate_strife_episode(reset_strength::Float64, community_effect::Float64; rng::AbstractRNG=Random.GLOBAL_RNG)
    # mean length scales with reset_strength; baseline ~8 cycles
    mean_len = 8.0 + 8.0*reset_strength
    # approximate sampling with Normal noise (avoids extra dependency); clamp to sensible range
    lenf = mean_len + randn(rng)*sqrt(max(mean_len,1.0))
    len = max(1, round(Int, clamp(lenf, 1.0, max(1.0, mean_len*4.0))))

    # optional ENV override to boost rebound probability (useful to model covenant/support)
    rebound_boost = try parse(Float64, get(ENV, "DAY8_REBOUND_BOOST", "0.0")) catch; 0.0 end

    # recovery prob depends on community, inverse of reset_strength, and optional boost
    rebound_prob = clamp(0.5 + 0.45*community_effect - 0.4*reset_strength + rebound_boost, 0.0, 1.0)
    rebound_success = rand(rng) < rebound_prob

    # fidelity change: successful rebounds produce boost; failed rebounds produce net loss
    # allow an ENV-controlled fidelity multiplier for stronger covenant-like renewal
    fidelity_boost_factor = try parse(Float64, get(ENV, "DAY8_FIDELITY_BOOST_FACTOR", "1.0")) catch; 1.0 end

    if rebound_success
        fidelity_delta = clamp((0.08 + 0.12*reset_strength) * fidelity_boost_factor, 0.01, 0.9)
    else
        fidelity_delta = -clamp((0.02 + 0.25*reset_strength) / max(fidelity_boost_factor, 0.01), 0.01, 0.9)
    end

    return (length=len, rebound_success=rebound_success, fidelity_delta=fidelity_delta, rebound_prob=rebound_prob)
end

function run_day8_eden_fall(prev_day6_path::AbstractString="day6_mc_results.csv"; trials::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)
    if !isfile(prev_day6_path)
        error("Day-6 results not found: " * prev_day6_path)
    end

    d6 = CSV.File(prev_day6_path) |> DataFrame
    N = nrow(d6)

    rows = Vector{Dict{String,Any}}()
    reset_count = 0
    total_strife_lengths = Int[]
    fidelity_changes = Float64[]
    rebound_failures = 0

    # ENV tunables for episode severity
    community_baseline = try parse(Float64, get(ENV, "DAY8_COMMUNITY_BASELINE", "0.8")) catch; 0.8 end

    for i in 1:trials
        prow = NamedTuple(d6[rand(1:N), :])
        # reuse perturbation_trial to detect reset and reset_strength
        p = perturbation_trial(prow)

        if p.reset_flag
            reset_count += 1
            # approximate community_effect from Day-6 flag
            community_effect = occursin("Cooperative", string(get(prow, :community_flag, ""))) ? 1.0 : community_baseline

            sp = simulate_strife_episode(p.reset_strength, community_effect)
            push!(total_strife_lengths, sp.length)
            push!(fidelity_changes, sp.fidelity_delta)
            if !sp.rebound_success
                rebound_failures += 1
            end
            push!(rows, Dict("trial"=>i, "reset_flag"=>true, "reset_strength"=>p.reset_strength, "strife_length"=>sp.length, "rebound_success"=>sp.rebound_success, "fidelity_delta"=>sp.fidelity_delta))
        else
            push!(rows, Dict("trial"=>i, "reset_flag"=>false, "reset_strength"=>0.0, "strife_length"=>0, "rebound_success"=>true, "fidelity_delta"=>0.0))
        end
    end

    reset_rate = reset_count / trials * 100
    mean_strife_length = isempty(total_strife_lengths) ? 0.0 : mean(total_strife_lengths)
    rebound_success_rate = reset_count == 0 ? 100.0 : (1.0 - rebound_failures/reset_count) * 100
    mean_fidelity_delta_pct = isempty(fidelity_changes) ? 0.0 : mean(fidelity_changes) * 100

    CSV.write("day8_eden_fall_results.csv", DataFrame(rows))

    # summary log
    open("day8_eden_fall_log.txt", "w") do io
        @printf(io, "Day-8 Eden/Fall — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Reset (Fall) rate: %.4f%%\n", reset_rate)
        @printf(io, "Mean strife episode length (cycles): %.3f\n", mean_strife_length)
        @printf(io, "Rebound success rate: %.3f%%\n", rebound_success_rate)
        @printf(io, "Mean fidelity delta (%%): %.3f\n", mean_fidelity_delta_pct)
    end

    return Dict("trials"=>trials, "reset_rate"=>reset_rate, "mean_strife_length"=>mean_strife_length, "rebound_success_rate"=>rebound_success_rate, "mean_fidelity_delta_pct"=>mean_fidelity_delta_pct)
end

# CLI
if abspath(PROGRAM_FILE) == @__FILE__
    res = run_day8_eden_fall("day6_mc_results.csv", trials=parse(Int, get(ENV, "DAY8_TRIALS", "10000")), rng_seed=42)
    println(res)
end
