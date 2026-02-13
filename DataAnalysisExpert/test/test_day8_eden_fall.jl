using Test
include("../day8_eden_fall.jl")

@testset "Day-8 Eden/Fall smoke" begin
    ENV["DAY8_TRIALS"] = "200"
    res = run_day8_eden_fall("day6/data/day6_mc_results.csv", trials=200, rng_seed=7)
    @test haskey(res, "reset_rate")
    @test res["reset_rate"] >= 0.0 && res["reset_rate"] <= 100.0
    @test isfile("day8_eden_fall_results.csv")
    delete!(ENV, "DAY8_TRIALS")
end
