# Runner for Day-8 narrative multi-epoch scenario
ENV["DAY7_PERTURB_SIGMA"] = get(ENV, "DAY7_PERTURB_SIGMA", "0.60")
ENV["DAY7_PERTURB_THRESHOLD"] = get(ENV, "DAY7_PERTURB_THRESHOLD", "1.74")
ENV["DAY7_COMMUNITY_DAMPING"] = get(ENV, "DAY7_COMMUNITY_DAMPING", "0.8")
ENV["DAY8_COMMUNITY_BASELINE"] = get(ENV, "DAY8_COMMUNITY_BASELINE", "1.10")
ENV["DAY8_REBOUND_BOOST"] = get(ENV, "DAY8_REBOUND_BOOST", "0.25")
ENV["DAY8_FIDELITY_BOOST_FACTOR"] = get(ENV, "DAY8_FIDELITY_BOOST_FACTOR", "1.15")
ENV["DAY8_NARRATIVE_EPOCHS"] = get(ENV, "DAY8_NARRATIVE_EPOCHS", "10000")

include("../day8_narrative.jl")
println("Running Day-8 narrative multi-epoch run -> writing day8_narrative_results.csv + fidelity history and plot")
res = run_day8_narrative("day6_mc_results.csv", epochs=parse(Int, ENV["DAY8_NARRATIVE_EPOCHS"]), rng_seed=42)
println(res)
# plot via Python helper
try
    run(`python tools/plot_day8_narrative.py day8_narrative_fidelity_history.csv day8_narrative_hist.png`)
    println("Wrote day8_narrative_hist.png")
catch err
    @warn "Failed to create Day-8 narrative plot" err
end