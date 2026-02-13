println("Running resampled 10k exoplanet MC + summarization...")
include(joinpath(@__DIR__, "..", "exoplanet_mc.jl"))
include(joinpath(@__DIR__, "..", "planetary_db.jl"))

archive_csv = joinpath(@__DIR__, "..", "data", "exoplanet_archive.csv")
out_csv = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k.csv")
summary_base = joinpath(@__DIR__, "..", "data", "planetary_mc_summary_10k")

# if archive CSV missing, fall back to sample
if !isfile(archive_csv)
    println("Archive CSV missing; falling back to sample CSV")
    archive_csv = joinpath(@__DIR__, "..", "data", "exoplanet_sample.csv")
end

run_resample_mc(archive_csv, out_csv; n=10000)
summary = summarize_mc_results(out_csv, summary_base; topn=10)
println("Summary:")
println(summary)
println("Generated:")
println(" - ", out_csv)
println(" - ", summary_base * ".classes.csv")
println(" - ", summary_base * ".top.csv")
println("Done.")
