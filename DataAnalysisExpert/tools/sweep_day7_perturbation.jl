# Sweep Day-7 perturbation sensitivity over sigma (jitter) and threshold for several community_damping levels
# Default design: 3 damping levels × 11×11 grid × 125 trials/cell -> ~45,375 total trials (ENV override available)

include(joinpath(@__DIR__, "..", "day7_perturbation.jl"))
using CSV, DataFrames, Printf

# configurable ranges (override via ENV)
sigma_min = parse(Float64, get(ENV, "DAY7_PERTURB_SIGMA_MIN", "0.05"))
sigma_max = parse(Float64, get(ENV, "DAY7_PERTURB_SIGMA_MAX", "0.60"))
sigma_steps = parse(Int, get(ENV, "DAY7_PERTURB_SIGMA_STEPS", "11"))
th_min = parse(Float64, get(ENV, "DAY7_PERTURB_THRESHOLD_MIN", "1.10"))
th_max = parse(Float64, get(ENV, "DAY7_PERTURB_THRESHOLD_MAX", "1.80"))
th_steps = parse(Int, get(ENV, "DAY7_PERTURB_THRESHOLD_STEPS", "11"))

# community damping levels to sweep (comma-separated ENV allowed)
comm_levels_str = get(ENV, "DAY7_COMM_DAMP_LEVELS", "0.4,0.8,1.2")
comm_levels = parse.(Float64, split(comm_levels_str, ','))

trials_per_cell = parse(Int, get(ENV, "DAY7_PERT_TRIALS_PER_CELL", "125"))
@printf("Day-7 perturbation sweep: %dx%d grid × %d damping levels, %d trials/cell -> ~%d total trials\n",
        sigma_steps, th_steps, length(comm_levels), trials_per_cell, sigma_steps*th_steps*length(comm_levels)*trials_per_cell)

sigmas = range(sigma_min, stop=sigma_max, length=sigma_steps)
ths = range(th_min, stop=th_max, length=th_steps)
rows = Vector{Dict{String,Any}}()

for cd in comm_levels
    for s in sigmas
        for t in ths
            ENV["DAY7_PERTURB_SIGMA"] = string(s)
            ENV["DAY7_PERTURB_THRESHOLD"] = string(t)
            ENV["DAY7_COMMUNITY_DAMPING"] = string(cd)
            ENV["DAY7_PERT_TRIALS"] = string(trials_per_cell)

            res = run_day7_perturbation("day6_mc_results.csv", trials=trials_per_cell, rng_seed=42)

            push!(rows, Dict("community_damping"=>cd, "sigma"=>s, "threshold"=>t, "trials"=>trials_per_cell, "reset_rate"=>res["reset_rate"], "rebound_success_rate"=>res["rebound_success_rate"], "mean_reset_strength"=>res["mean_reset_strength"], "mean_fidelity_change_pct"=>res["mean_fidelity_change_pct"]))

            # cleanup ENV keys used for single-cell run
            for k in ("DAY7_PERTURB_SIGMA","DAY7_PERTURB_THRESHOLD","DAY7_COMMUNITY_DAMPING","DAY7_PERT_TRIALS")
                if haskey(ENV, k)
                    delete!(ENV, k)
                end
            end
        end
    end
end

out = joinpath(@__DIR__, "..", "sensitivity", "day7_perturbation_sweep.csv")
mkpath(joinpath(@__DIR__, "..", "sensitivity"))
CSV.write(out, DataFrame(rows))
println("Wrote sweep CSV: ", out)

# plotting via Python helper (non-fatal)
png_out = joinpath(@__DIR__, "..", "sensitivity", "day7_perturbation_sweep_heatmaps.png")
try
    run(`python tools/plot_day7_perturbation_sweep.py $(out) $(png_out)`)
    println("Wrote sweep heatmaps: ", png_out)
catch err
    @warn "Failed to create perturbation sweep plots" err
end

println("Day-7 perturbation sweep complete.")
