using Test
include("../day4.jl")

@testset "Day 4 — chemistry & thermal integration" begin
    # chemical_proxy yields a finite fraction between 0 and 1
    c = chemical_proxy(BigFloat(1e60))
    @test 0.0 <= c <= 1.0

    # thermal_convection in [0,1]
    tc = thermal_convection(BigFloat(1.2), 95.0)
    @test 0.0 <= tc <= 1.0

    # rna_to_dna behavior: low convection -> folding false, high convection + gens -> true
    r1 = rna_to_dna(BigFloat(4.0), 0.1, 1e9)
    @test r1.folding == false
    r2 = rna_to_dna(BigFloat(4.0), 0.9, 1e9)
    @test r2.folding == true
    @test r2.complexity > 1.0

    # run_day4 should produce day4_state.json and include the new keys
    st = run_day4()
    @test isfile("day4_state.json")
    @test haskey(st, "chemical_CN_O_frac")
    @test haskey(st, "convection_efficiency")
    @test haskey(st, "rna_to_dna_folding")
end
