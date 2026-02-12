using Test
include("../day6.jl")

@testset "Day 6 — land animals & humanity scaffold" begin
    st = run_day6()
    @test haskey(st, "day")
    if isa(st["day"], Integer)
        @test Int(st["day"]) == 6
    else
        @test parse(Int, st["day"]) == 6
    end
    @test haskey(st, "land_animals")
    @test haskey(st, "humanity_image")
    @test isfile("day6_state.json")
end
