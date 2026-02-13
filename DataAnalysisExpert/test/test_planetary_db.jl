using Test

include("../day3/planetary_db.jl")
include("../day3/exoplanet_mc.jl")

@testset "Planetary DB & Exoplanet MC" begin
    sample = joinpath(@__DIR__, "..", "data", "exoplanet_sample.csv")
    rows = load_exoplanet_csv(sample)
    @test length(rows) == 5

    cls1 = classify_planet(rows[1])
    @test cls1[:is_rocky] == true
    @test cls1[:low_e] == true
    @test cls1[:em_retention] >= 0.0 && cls1[:em_retention] <= 1.0

    stats = sample_planetary_stats(rows)
    @test isfinite(stats[:r_mean])
    @test stats[:r_min] <= stats[:r_max]

    # run a tiny MC and validate output CSV is produced and contains habitability_score
    out = joinpath(@__DIR__, "..", "planetary_mc_results_sample.csv")
    _ = run_exoplanet_mc(sample, out; n=10)
    @test isfile(out)
    tbl = CSV.File(out) |> DataFrame
    @test any(x->(x == :habitability_score) || (x == "habitability_score"), names(tbl))
    # access column by string/symbol robustly
    colname = any(x->x==:habitability_score, names(tbl)) ? :habitability_score : "habitability_score"
    vals = collect(tbl[!, colname])
    @test all((x->x >= 0.0 && x <= 1.0).(vals))

    # test query builder for Exoplanet Archive (no network)
    q = build_exoplanet_archive_query()
    @test typeof(q) == String
    @test occursin("TAP/sync", q)

    # cleanup
    rm(out; force=true)
end
