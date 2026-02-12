using Test
include("../day3.jl")
include("../day3_mc.jl")

@testset "Day 3 — clamping & MC behavior" begin
    # compute_S_day3 clamps: very large seed should not produce Inf and should be within clamp bounds
    large_seed = BigFloat(1e30)
    s_large = compute_S_day3(large_seed, BigFloat(7))
    @test isfinite(Float64(s_large))
    @test s_large <= BigFloat(250.0) && s_large >= BigFloat(-100.0)

    # vegetation proxy: S <= 3 -> vegetation == 0
    s_low = BigFloat(2.5)
    res_low = anchor_to_matter_veg(s_low, 1.0)
    @test res_low.condense == false
    @test res_low.vegetation == 0.0

    # S > 3 -> vegetation > 0 and finite
    s_high = BigFloat(4.2)
    res_high = anchor_to_matter_veg(s_high, 1e3)
    @test res_high.condense == true
    @test isfinite(res_high.vegetation) && res_high.vegetation > 0.0

    # run a small MC and validate outputs exist & summary metrics in sensible range
    summary = run_day3_mc(trials=50, rng_seed=7)
    @test haskey(summary, "success_rate")
    @test 0.0 <= summary["success_rate"] <= 100.0
    # result CSV/log should have been created by the MC run
    @test isfile("day3_mc_results.csv")
    @test isfile("day3_mc_log.txt")
end
