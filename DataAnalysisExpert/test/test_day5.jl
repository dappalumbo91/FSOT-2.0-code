using Test
include("../day1/day1.jl")
include("../day5.jl")

@testset "Day 5 — swarming biota scaffold" begin
    st = run_day5()
    @test haskey(st, "day")
    if isa(st["day"], Integer)
        @test Int(st["day"]) == 5
    else
        @test parse(Int, st["day"]) == 5
    end
    @test haskey(st, "swarming")
    @test haskey(st, "diversity_proxy")
    @test isfile("day5_state.json")
end
