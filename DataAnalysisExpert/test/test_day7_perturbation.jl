using Test
include("../day7_perturbation.jl")

@testset "Day-7 perturbation smoke" begin
    ENV["DAY7_PERTURB_SIGMA"] = "0.5"
    res = run_day7_perturbation("day6/data/day6_mc_results.csv", trials=200, rng_seed=7)
    @test haskey(res, "reset_rate")
    @test res["reset_rate"] >= 0.0 && res["reset_rate"] <= 100.0
    @test isfile("day7_perturbation_results.csv")
    delete!(ENV, "DAY7_PERTURB_SIGMA")
end
