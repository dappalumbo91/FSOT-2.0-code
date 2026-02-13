include(joinpath(@__DIR__, "..", "planetary_db.jl"))
println("Re-downloading Exoplanet Archive with RA/DEC/hostname columns...")
fetch_exoplanet_archive_csv("data/exoplanet_archive.csv")
println("Done: data/exoplanet_archive.csv")
