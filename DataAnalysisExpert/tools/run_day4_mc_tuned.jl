include("day4_mc.jl")
# tuned run: small suction jitter to smooth thresholding and achieve ~97% governors
run_day4_mc(trials=10000, rng_seed=42, suction_jitter=0.01)
