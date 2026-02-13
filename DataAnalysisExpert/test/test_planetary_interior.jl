# Tests for the new planetary interior / Earth-baseline helpers
using Test
include("../day3/planetary_db.jl")

@testset "planetary_db: earth baseline" begin
    b = get_earth_baseline()
    @test isa(b, Dict)
    @test haskey(b, :tectonic_rate)
    @test b[:ocean_fraction] ≈ 0.71 atol=1e-3
    @test b[:dynamo_scale] == 1.0
end

# quick integration: run day3 small and assert new columns appear
include("../day3/day3_mc.jl")
res = run_day3_mc(trials=10, rng_seed=7)
@test isfile("day3_mc_results.csv")
lines = readlines("day3_mc_results.csv")
@test occursin("tectonic_rate", lines[1])
@test occursin("ocean_fraction", lines[1])
