println("Applying Gaia enrichment to 10k MC and summarizing...")
include(joinpath(@__DIR__, "..", "planetary_gaia.jl"))
include(joinpath(@__DIR__, "..", "exoplanet_mc.jl"))

mc_in = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k.csv")
gaia_in = joinpath(@__DIR__, "..", "data", "exoplanet_archive_gaia_sample.csv")
mc_out = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k_enriched.csv")

if !isfile(gaia_in)
    error("Gaia enrichment file not found: ", gaia_in)
end
apply_gaia_enrichment(mc_in, gaia_in, mc_out)
summary = summarize_mc_results(mc_out, joinpath(@__DIR__, "..", "data", "planetary_mc_summary_10k_enriched"); topn=10)
println("Enrichment summary:")
println(summary)
println("Wrote enriched MC: ", mc_out)
println("Done.")
