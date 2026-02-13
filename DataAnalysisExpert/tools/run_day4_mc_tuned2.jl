include("day4_mc.jl")
# tuned run: set base_suction to 3.505 and small jitter to get governors ~97-98%
run_day4_mc(trials=10000, rng_seed=42, base_suction=3.52, suction_jitter=0.005)
