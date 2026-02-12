using Test
include("../day6_mc.jl")

@testset "Day 6 MC — humanity stability checks" begin
    summary = run_day6_mc(trials=200, rng_seed=7)
    @test haskey(summary, "land_rate") && 0.0 <= summary["land_rate"] <= 100.0
    @test haskey(summary, "humanity_rate") && 0.0 <= summary["humanity_rate"] <= 100.0
    @test isfile("day6_mc_results.csv")
    @test isfile("day6_mc_log.txt")
end
