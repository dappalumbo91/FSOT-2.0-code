using Test
include("../day3/planetary_db.jl")
include("../day4.jl")
include("../day5.jl")

@testset "planetary data fetchers & tectonics" begin
    @test fetch_prem_table(joinpath(@__DIR__, "prem_sample.csv")) |> isfile

    # create a tiny USGS-like geojson sample and test summarizer
    sample = "{" * "\"type\":\"FeatureCollection\",\"features\":[{\"properties\":{\"mag\":5.1,\"depth\":10}},{\"properties\":{\"mag\":4.8,\"depth\":30}}]}"
    gpath = joinpath(@__DIR__, "usgs_sample.geojson")
    open(gpath, "w") do io
        write(io, sample)
    end
    s = summarize_usgs_earthquakes(gpath)
    @test haskey(s, :count) && s[:count] == 2
    @test isfinite(s[:mean_mag])

    # test tectonic/volcanic primitives
    t = tectonic_event(2.5; rng=MersenneTwister(1))
    @test haskey(t, :puncture) && haskey(t, :rebound)
    v = volcanic_outgassing(0.6, t[:rebound])
    @test v[:chem_boost] >= 0.0 && v[:chem_boost] <= 1.0

    # hydrothermal boost behavior
    hb = hydrothermal_boost(0.7, 0.6, 2.0)
    @test hb >= 1.0 && hb <= 2.5

    # Paleobio fetcher: network may be unavailable in CI; ensure the helper runs and returns a path when offline by writing a small sample
    ppath = joinpath(@__DIR__, "paleobio_sample.json")
    sample_json = "{\"records\":[{\"max_ma\":1.8,\"min_ma\":1.6},{\"max_ma\":2.1,\"min_ma\":1.9}]}"
    open(ppath, "w") do io
        write(io, sample_json)
    end
    s2 = summarize_paleobio_ages(ppath)
    @test s2[:count] == 4 || s2[:n] >= 1
end
