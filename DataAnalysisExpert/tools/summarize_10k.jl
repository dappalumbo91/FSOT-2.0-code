include(joinpath(@__DIR__, "..", "exoplanet_mc.jl"))
println("Summarizing 10k MC results...")
summary = summarize_mc_results("data/planetary_mc_results_10k.csv", "data/planetary_mc_summary_10k"; topn=10)
println(summary)
