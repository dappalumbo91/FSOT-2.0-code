# Day-8 Narrative — multi-epoch Eden/Fall timeline generator
# Chains calibrated Day-7 perturbations + Day-8 strife/rebound to produce epoch timelines

include("day8_eden_fall.jl")
include("day7_perturbation.jl")
using CSV, DataFrames, Random, Statistics, Printf, Dates

function run_day8_narrative(prev_day6_path::AbstractString="day6_mc_results.csv"; epochs::Int=10000, rng_seed::Int=42)
    Random.seed!(rng_seed)
    if !isfile(prev_day6_path)
        error("Day-6 results not found: " * prev_day6_path)
    end

    d6 = CSV.File(prev_day6_path) |> DataFrame
    N = nrow(d6)

    rows = Vector{Dict{String,Any}}()
    fidelity = 1.0
    fidelity_history = Float64[]
    epoch_reset_count = 0
    total_strife_lengths = Int[]
    fidelity_changes = Float64[]
    rebound_failures = 0

    start_ts = now()
    for e in 1:epochs
        prow = NamedTuple(d6[rand(1:N), :])
        p = perturbation_trial(prow)
        fidelity_before = fidelity
        if p.reset_flag
            epoch_reset_count += 1
            # community effect as in Day-8
            community_effect = occursin("Cooperative", string(get(prow, :community_flag, ""))) ? 1.0 : try parse(Float64, get(ENV, "DAY8_COMMUNITY_BASELINE", "0.8")) catch; 0.8 end
            sp = simulate_strife_episode(p.reset_strength, community_effect)
            if sp.rebound_success
                fidelity *= (1.0 + sp.fidelity_delta)
            else
                fidelity *= (1.0 + sp.fidelity_delta) # fidelity_delta may be negative
                rebound_failures += 1
            end
            push!(total_strife_lengths, sp.length)
            push!(fidelity_changes, sp.fidelity_delta)
            push!(rows, Dict("epoch"=>e, "reset_flag"=>true, "reset_strength"=>p.reset_strength, "strife_length"=>sp.length, "rebound_success"=>sp.rebound_success, "fidelity_before"=>fidelity_before, "fidelity_after"=>fidelity))
        else
            # small baseline fidelity drift (optional)
            fidelity *= 1.00001
            push!(rows, Dict("epoch"=>e, "reset_flag"=>false, "reset_strength"=>0.0, "strife_length"=>0, "rebound_success"=>true, "fidelity_before"=>fidelity_before, "fidelity_after"=>fidelity))
        end
        push!(fidelity_history, fidelity)
    end
    elapsed = now() - start_ts

    reset_rate = epoch_reset_count / epochs * 100
    mean_strife_length = isempty(total_strife_lengths) ? 0.0 : mean(total_strife_lengths)
    rebound_success_rate = epoch_reset_count == 0 ? 100.0 : (1.0 - rebound_failures/epoch_reset_count) * 100
    cumulative_fidelity_change_pct = (fidelity / 1.0 - 1.0) * 100

    CSV.write("day8_narrative_results.csv", DataFrame(rows))

    open("day8_narrative_log.txt", "w") do io
        @printf(io, "Day-8 narrative — epochs=%d, seed=%d\n", epochs, rng_seed)
        @printf(io, "Elapsed time: %s\n", string(elapsed))
        @printf(io, "Epoch reset rate: %.4f%%\n", reset_rate)
        @printf(io, "Mean strife episode length (cycles): %.3f\n", mean_strife_length)
        @printf(io, "Rebound success rate: %.3f%%\n", rebound_success_rate)
        @printf(io, "Cumulative fidelity change (%%): %.3f\n", cumulative_fidelity_change_pct)
    end

    # write fidelity history for plotting
    CSV.write("day8_narrative_fidelity_history.csv", DataFrame(epoch=1:epochs, fidelity=fidelity_history))

    return Dict("epochs"=>epochs, "reset_rate"=>reset_rate, "mean_strife_length"=>mean_strife_length, "rebound_success_rate"=>rebound_success_rate, "cumulative_fidelity_change_pct"=>cumulative_fidelity_change_pct)
end

if abspath(PROGRAM_FILE) == @__FILE__
    res = run_day8_narrative("day6_mc_results.csv", epochs=parse(Int, get(ENV, "DAY8_NARRATIVE_EPOCHS", "10000")), rng_seed=42)
    println(res)
end
