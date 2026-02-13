ENV["DAY7_CURIO_FEEDBACK_K"] = "4.0"
ENV["DAY7_CURIO_SHARPNESS"] = "2.0"
ENV["DAY7_CURIO_WEIGHT"] = "0.8"
ENV["DAY7_CYCLE_BASELINE"] = "0.25"
ENV["DAY7_COMMUNITY_BASELINE"] = "0.5"

include("../day7.jl")
run_day7("day6_mc_results.csv", trials=2000, rng_seed=42)
