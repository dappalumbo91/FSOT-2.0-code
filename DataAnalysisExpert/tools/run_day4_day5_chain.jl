# Chain Day-4 and Day-5 MC runs using the Gaia-enriched Day-3 output (uses new tectonics/ocean/weather priors)
include(joinpath(@__DIR__, "..", "day4_mc.jl"))
include(joinpath(@__DIR__, "..", "day5", "day5_mc.jl"))

println("Running Day-4 MC (10k trials) using day3/data/day3_mc_results_gaia.csv")
res4 = run_day4_mc(trials=10000, rng_seed=42, day3_path="day3/data/day3_mc_results_gaia.csv")
println("Day-4 summary:")
println(res4)

println("Running Day-5 MC (10k trials) — will pick up ocean/weather boosts from Day-4 summary")
res5 = run_day5_mc(trials=10000, rng_seed=42)
println("Day-5 summary:")
println(res5)
