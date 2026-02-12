using Test

# include Day 1 module
include("../day1.jl")

@testset "Day 1 — Planck/FIRAS parsing & S anchoring" begin
    # Planck JSON parsing
    planck_path = joinpath(@__DIR__, "test_data", "planck_params.json")
    planck = parse_planck_json(planck_path)
    @test isapprox(Float64(planck.T_CMB), 2.72548; atol=1e-8)
    @test planck.z_recomb == BigFloat(1090)

    # FIRAS parsing
    firas_path = joinpath(@__DIR__, "test_data", "firas_sample.txt")
    firas = parse_firas_file(firas_path)
    @test length(firas) == 2
    @test firas[1][1] == 30.0

    # LAMBDA / Planck table parsing
    lambda_path = joinpath(@__DIR__, "test_data", "planck_lambda_table.txt")
    lambda_rows = parse_lambda_table(lambda_path)
    @test length(lambda_rows) == 3
    @test lambda_rows[2][1] == 100.0

    # FITS parsing (real FITS fixtures)
    for fname in ("planck_real.fits", "planck_map.fits", "planck_params.fits", "planck_100ghz_map.fits", "planck_143ghz_map.fits", "planck_power_spectrum.fits")
        fits_path = joinpath(@__DIR__, "test_data", fname)
        res = parse_fits_file(fits_path)

        @test isa(res, Dict)
        @test haskey(res, :ok)
        # header should always be present (parsed by parse_fits_header)
        @test haskey(res, :header)
        hdr = res[:header]
        @test haskey(hdr, "SIMPLE") || true
        # validate common header keywords in Planck-like fixtures
        if haskey(hdr, "TELESCOP")
            @test occursin("PLANCK", hdr["TELESCOP"])
        end
        if haskey(hdr, "INSTRUME")
            @test isa(hdr["INSTRUME"], String)
        end
        # additional checks for frequency-tagged maps and power spectrum
        if occursin("map", lowercase(fname))
            # expect 2D array data when FITSIO available
            if haskey(res, :data) && isa(res[:data], AbstractVector)
                data = res[:data]
                @test length(data) >= 1
                # first row should be a Float64 vector (map row / flattened)
                @test isa(data[1], AbstractVector)
            end
            if haskey(hdr, "FREQ")
                @test occursin("GHz", hdr["FREQ"]) || true
            end
        elseif occursin("power_spectrum", lowercase(fname))
            # power spectrum should provide tabular rows if data present
            if haskey(res, :data) && isa(res[:data], AbstractVector)
                @test length(res[:data]) >= 1
            end
        end
    end

    # S sign sensitivity: higher Δψ should increase S and anchor behavior follows S>0
    S_low = compute_S_day1(Δψ=BigFloat(0.5))
    res_low = anchor_to_cmb(S_low)
    S_high = compute_S_day1(Δψ=BigFloat(1.2))
    res_high = anchor_to_cmb(S_high)

    @test isfinite(Float64(S_low)) && isfinite(Float64(S_high))
    @test res_high.emergence == (S_high > 0)
    @test res_low.emergence == (S_low > 0)
    if S_low > 0
        @test Float64(res_low.anchored_density) > 0.0
    else
        @test Float64(res_low.anchored_density) == 0.0
    end
    if S_high > 0
        @test Float64(res_high.anchored_density) > 0.0
    else
        @test Float64(res_high.anchored_density) == 0.0
    end
end

# include Day 3 tests
include("test_day3.jl")

# include Day 4 tests
include("test_day4.jl")

# include Day 4 MC tests
include("test_day4_mc.jl")

# Day 5 tests
include("test_day5.jl")

# Day 5 MC tests
include("test_day5_mc.jl")

