println("Running Gaia cross-match (limited, network required)...")
include(joinpath(@__DIR__, "..", "planetary_gaia.jl"))
archive_csv = joinpath(@__DIR__, "..", "data", "exoplanet_archive.csv")
out_csv = joinpath(@__DIR__, "..", "data", "exoplanet_archive_gaia_sample.csv")

# limit to first 500 unique hosts to avoid rate issues
crossmatch_archive_with_gaia(archive_csv, out_csv; limit=500)
println("Wrote: ", out_csv)
println("Done.")
