using Test
include("../day8_narrative.jl")

@testset "Day-8 narrative smoke" begin
    ENV["DAY8_NARRATIVE_EPOCHS"] = "200"
    res = run_day8_narrative("day6/data/day6_mc_results.csv", epochs=200, rng_seed=7)
    @test haskey(res, "reset_rate")
    @test res["reset_rate"] >= 0.0 && res["reset_rate"] <= 100.0
    @test isfile("day8/data/day8_narrative_results.csv")
    delete!(ENV, "DAY8_NARRATIVE_EPOCHS")
end
