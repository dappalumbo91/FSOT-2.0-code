using Test
include("../day4/day4_mc.jl")

@testset "Day 4 MC — stability checks" begin
    summary = run_day4_mc(trials=200, rng_seed=7)
    @test haskey(summary, "gov_rate")
    @test 0.0 <= summary["gov_rate"] <= 100.0
    @test haskey(summary, "fold_rate")
    @test 0.0 <= summary["fold_rate"] <= 100.0
    @test isfile("day4_mc_summary.csv")
    @test isfile("day4_mc_log.txt")
end
