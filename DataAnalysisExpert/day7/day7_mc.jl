include(joinpath(@__DIR__, "..", "day7.jl"))

println("Running Day-7 MC (defaults: trials=2000 for quick run)")
res = run_day7("day6_mc_results.csv", trials=parse(Int, get(ENV, "DAY7_TRIALS", "2000")), rng_seed=42)
println("Day-7 MC result:")
println(res)
println("Wrote: day7_mc_results.csv, day7_mc_log.txt")
