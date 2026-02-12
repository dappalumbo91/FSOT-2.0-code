using Test
include("../day5_mc.jl")

@testset "Day 5 MC — biota stability & diversity" begin
    summary = run_day5_mc(trials=200, rng_seed=7)
    @test haskey(summary, "swarming_rate")
    @test 0.0 <= summary["swarming_rate"] <= 100.0
    @test isfile("day5_mc_results.csv")
    @test isfile("day5_mc_log.txt")
end
