using Test
include("../day6/day6_mc.jl")

@testset "Day-6 hominid calibration" begin
    res = run_day6_mc(trials=200, rng_seed=7)
    @test isfile("day6_mc_hominid.csv")
    lines = readlines("day6_mc_hominid.csv")
    @test occursin("divergence_Ma", lines[1])
    # check at least one sapiens_flag present or NaN entries allowed
    df = CSV.File("day6_mc_hominid.csv") |> DataFrame
    @test :sapiens_Ma in names(df)
    # ensure returned summary contains humanity_rate
    @test haskey(res, "humanity_rate")
end
