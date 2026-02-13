# Runner for Day-8 Eden/Fall MC (uses tuned Day-7 defaults / perturbation tails)
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
ENV["DAY8_COMMUNITY_BASELINE"] = get(ENV, "DAY8_COMMUNITY_BASELINE", "0.8")
ENV["DAY8_TRIALS"] = get(ENV, "DAY8_TRIALS", "10000")

include("../day8_eden_fall.jl")
println("Running Day-8 Eden/Fall MC (tuned Day-7 sweet-spot + perturbation tails)")
res = run_day8_eden_fall("day6_mc_results.csv", trials=parse(Int, ENV["DAY8_TRIALS"]), rng_seed=42)
println(res)
# plot results via Python helper (non-fatal)
try
    run(`python tools/plot_day8_eden_fall.py day8_eden_fall_results.csv day8_eden_fall_hist.png`)
    println("Wrote day8_eden_fall_hist.png")
catch err
    @warn "Failed to create Day-8 histogram with Python" err
end