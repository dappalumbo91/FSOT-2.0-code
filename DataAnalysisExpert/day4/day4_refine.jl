using Printf, Statistics
include("day4_mc.jl")

function refine(; trials=3000)
    sm_list = 3.49:0.002:3.52
    best = nothing
    best_diff = 1e9
    for sm in sm_list
        govs = 0
        for i in 1:trials
            S = compute_S_day4_mc(BigFloat(3.41), BigFloat(5); delta_psi_base=-1.8, suction_mult=sm)
            govs += (Float64(S) < 0.0) ? 1 : 0
        end
        rate = govs / trials * 100
        diff = abs(rate - 97.5)
        @printf("sm=%.4f -> gov_rate=%.3f%%\n", sm, rate)
        if diff < best_diff
            best_diff = diff
            best = (sm, rate)
        end
    end
    return best
end

refine(trials=3000)
