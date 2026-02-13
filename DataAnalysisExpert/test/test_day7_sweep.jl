using Test
include("../day7.jl")
include("../tools/sweep_day7_rest.jl")

@testset "Day-7 sweep smoke" begin
    # run a tiny sweep (override trials per cell) for CI-friendly test
    ENV["DAY7_SWEEP_TRIALS_PER_CELL"] = "10"
    ENV["DAY7_CURIOSITY_STEPS"] = "3"
    ENV["DAY7_COMMUNITY_STEPS"] = "3"
    ENV["DAY7_CURIOSITY_MIN"] = "0.6"; ENV["DAY7_CURIOSITY_MAX"] = "1.4"
    ENV["DAY7_COMMUNITY_MIN"] = "0.8"; ENV["DAY7_COMMUNITY_MAX"] = "1.2"
    ENV["DAY7_CURIO_FEEDBACK_K"] = "2.0"
    run(`julia --project=. tools/sweep_day7_rest.jl`)
    @test isfile("sensitivity/day7_rest_sweep.csv")
    # basic validation: CSV contains expected columns
    df = CSV.File("sensitivity/day7_rest_sweep.csv") |> DataFrame
    @test :cycle_rate in names(df) && :rest_rate in names(df)
    for k in ("DAY7_SWEEP_TRIALS_PER_CELL","DAY7_CURIOSITY_STEPS","DAY7_COMMUNITY_STEPS","DAY7_CURIOSITY_MIN","DAY7_CURIOSITY_MAX","DAY7_COMMUNITY_MIN","DAY7_COMMUNITY_MAX","DAY7_CURIO_FEEDBACK_K")
        if haskey(ENV, k)
            delete!(ENV, k)
        end
    end
end
