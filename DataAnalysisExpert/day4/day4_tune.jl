# Quick tuner for Day-4 governor percentage
using Random, Statistics, Printf
include("day4_mc.jl")

function sweep_tune(; trials_per_combo=2000)
    deltas = [-1.8]
    suctions = [3.45, 3.48, 3.50, 3.52, 3.55]
    results = []
    for dp in deltas, sm in suctions
        govs = 0
        for i in 1:trials_per_combo
            S = compute_S_day4_mc(BigFloat(3.41), BigFloat(5); delta_psi_base=dp, suction_mult=sm)
            if Float64(S) < 0.0
                govs += 1
            end
        end
        rate = govs / trials_per_combo * 100
        push!(results, (delta=dp, suction=sm, gov_rate=rate))
        @printf("dp=%.2f sm=%.3f -> gov_rate=%.2f%%\n", dp, sm, rate)
    end
    return results
end

sweep_tune(trials_per_combo=2000)
