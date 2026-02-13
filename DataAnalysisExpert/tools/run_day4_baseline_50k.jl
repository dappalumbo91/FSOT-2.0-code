include(joinpath(@__DIR__, "..", "day4_mc.jl"))
res = run_day4_mc(trials=50000, rng_seed=42, base_suction=3.5, suction_jitter=0.005, day3_path="data/day3_mc_results_gaia.csv")
println(res)
