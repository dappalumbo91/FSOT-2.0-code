using Test
include("../tools/sweep_day7_perturbation.jl")

@testset "Day-7 perturbation sweep smoke" begin
    # override for CI: tiny grid + tiny trials
    ENV["DAY7_PERTURB_SIGMA_MIN"] = "0.1"; ENV["DAY7_PERTURB_SIGMA_MAX"] = "0.2"; ENV["DAY7_PERTURB_SIGMA_STEPS"] = "3"
    ENV["DAY7_PERTURB_THRESHOLD_MIN"] = "1.2"; ENV["DAY7_PERTURB_THRESHOLD_MAX"] = "1.3"; ENV["DAY7_PERTURB_THRESHOLD_STEPS"] = "3"
    ENV["DAY7_COMM_DAMP_LEVELS"] = "0.5,0.8"
    ENV["DAY7_PERT_TRIALS_PER_CELL"] = "10"

    run(`julia --project=. tools/sweep_day7_perturbation.jl`)
    @test isfile("sensitivity/day7_perturbation_sweep.csv")

    for k in ("DAY7_PERTURB_SIGMA_MIN","DAY7_PERTURB_SIGMA_MAX","DAY7_PERTURB_SIGMA_STEPS","DAY7_PERTURB_THRESHOLD_MIN","DAY7_PERTURB_THRESHOLD_MAX","DAY7_PERTURB_THRESHOLD_STEPS","DAY7_COMM_DAMP_LEVELS","DAY7_PERT_TRIALS_PER_CELL")
        if haskey(ENV, k)
            delete!(ENV, k)
        end
    end
end
