# Download the Exoplanet Archive (ps table) and run a habitability MC (uses exoplanet_mc.jl)
println("Exoplanet Archive run starting...")
include(joinpath(@__DIR__, "..", "exoplanet_mc.jl"))
try
    out = run_exoplanet_archive_mc("data/planetary_mc_results_archive.csv"; download_path="data/exoplanet_archive.csv", n=5000)
    println("MC output written to: ", out)
catch e
    println("Exoplanet Archive download or MC failed: ", e)
    rethrow(e)
end
println("Done.")
