# Helper to run Day-7 perturbation experiment with tuned sweet-spot ENV presets
ENV["DAY7_CURIOSITY_SCALE"] = get(ENV, "DAY7_CURIOSITY_SCALE", "1.125")
ENV["DAY7_COMMUNITY_SCALE"] = get(ENV, "DAY7_COMMUNITY_SCALE", "1.125")
ENV["DAY7_CURIO_FEEDBACK_K"] = get(ENV, "DAY7_CURIO_FEEDBACK_K", "4.0")
ENV["DAY7_CURIO_SHARPNESS"] = get(ENV, "DAY7_CURIO_SHARPNESS", "2.0")
ENV["DAY7_CURIO_WEIGHT"] = get(ENV, "DAY7_CURIO_WEIGHT", "1.2")
ENV["DAY7_COMMUNITY_BASELINE"] = get(ENV, "DAY7_COMMUNITY_BASELINE", "0.8")
ENV["DAY7_CYCLE_BASELINE"] = get(ENV, "DAY7_CYCLE_BASELINE", "0.20")
ENV["DAY7_S_WEIGHT"] = get(ENV, "DAY7_S_WEIGHT", "0.15")
ENV["DAY7_PERTURB_SIGMA"] = get(ENV, "DAY7_PERTURB_SIGMA", "0.25")
ENV["DAY7_PERTURB_THRESHOLD"] = get(ENV, "DAY7_PERTURB_THRESHOLD", "1.4")
ENV["DAY7_COMMUNITY_DAMPING"] = get(ENV, "DAY7_COMMUNITY_DAMPING", "0.8")
ENV["DAY7_PERT_TRIALS"] = get(ENV, "DAY7_PERT_TRIALS", "10000")

include("../day7_perturbation.jl")
println("Running Day-7 perturbation experiment (tuned sweet-spot) -> writing day7_perturbation_results.csv + day7_perturbation_hist.png")
res = run_day7_perturbation("day6_mc_results.csv", trials=parse(Int, ENV["DAY7_PERT_TRIALS"]), rng_seed=42)
println(res)
# try to create histogram via Python helper (non-fatal)
try
    run(`python tools/plot_day7_perturbation.py day7_perturbation_results.csv day7_perturbation_hist.png`)
    println("Wrote day7_perturbation_hist.png")
catch err
    @warn "Failed to create perturbation histogram with Python" err
end
