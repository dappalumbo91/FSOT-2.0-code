# Quick integration test for the sensitivity sweep
# - runs the sweep in tiny-mode (fast)
# - asserts the summary CSV exists and has expected columns

using Test, CSV, DataFrames

# run sweep with very small trial count to keep CI fast
ENV["SWEEP_TRIALS"] = "4"
ENV["SWEEP_EXTENDED"] = "false"
include(joinpath(@__DIR__, "..", "tools", "sensitivity_sweep.jl"))

out = joinpath(@__DIR__, "..", "sensitivity", "sensitivity_sweep_summary.csv")
@test isfile(out)

df = CSV.File(out) |> DataFrame
# expected columns (subset)
expected = [:multiplier, :day4_gov_rate, :day4_habitable_frac, :day6_human_rate, :mean_dominion]
for c in expected
    @test c in names(df)
end

# minimal sanity checks
@test nrow(df) > 0
@test all(!ismissing, df.multiplier)

# now run a tiny extended sweep (metallicity->dynamo axis) and assert dynamo_scale present
ENV["SWEEP_TRIALS"] = "2"
ENV["SWEEP_EXTENDED"] = "true"
include(joinpath(@__DIR__, "..", "tools", "sensitivity_sweep.jl"))
df2 = CSV.File(out) |> DataFrame
@test :dynamo_scale in names(df2)
@test nrow(df2) >= 2
