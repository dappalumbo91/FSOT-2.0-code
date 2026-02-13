using Test
include("../day7.jl")

@testset "Day-7 rest equilibrium" begin
    res = run_day7("day6/data/day6_mc_results.csv", trials=200, rng_seed=7)
    @test haskey(res, "rest_rate")
    @test res["rest_rate"] >= 0.0 && res["rest_rate"] <= 100.0
    @test isfile("day7/data/day7_mc_results.csv")

    # smoke test with scaled curiosity/community params
    ENV["DAY7_CURIOSITY_SCALE"] = "1.5"
    ENV["DAY7_COMMUNITY_SCALE"] = "1.2"
    res2 = run_day7("day6/data/day6_mc_results.csv", trials=100, rng_seed=13)
    @test haskey(res2, "cycle_rate")
    delete!(ENV, "DAY7_CURIOSITY_SCALE"); delete!(ENV, "DAY7_COMMUNITY_SCALE")
end
