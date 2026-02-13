using Test
include("../day2.jl")

@testset "Day 2 basic behavior" begin
    # Ensure Day 2 can run using existing day1_state.json
    state = run_day2(save_state=false)
    @test haskey(state, "S")
    @test haskey(state, "post_D_eff")
    # S should be finite
    s = parse(Float64, state["S"])
    @test isfinite(s)
end
