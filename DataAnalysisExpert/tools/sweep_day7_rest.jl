# Sweep Day-7 rest stability over curiosity_scale and community_strength
# Grid: curiosity_scale ∈ [0.5,1.5] (±50%), community_strength ∈ [0.7,1.3] (±30%)
# Default total trials ≈ 50k divided across grid (5x5 grid -> 2000 trials per cell)

include(joinpath(@__DIR__, "..", "day7.jl"))
using CSV, DataFrames, Printf

# configurable grid size and ranges (override via ENV)
cs_min = parse(Float64, get(ENV, "DAY7_CURIOSITY_MIN", "0.4"))
cs_max = parse(Float64, get(ENV, "DAY7_CURIOSITY_MAX", "1.6"))
cs_steps = parse(Int, get(ENV, "DAY7_CURIOSITY_STEPS", "7"))
cm_min = parse(Float64, get(ENV, "DAY7_COMMUNITY_MIN", "0.6"))
cm_max = parse(Float64, get(ENV, "DAY7_COMMUNITY_MAX", "1.4"))
cm_steps = parse(Int, get(ENV, "DAY7_COMMUNITY_STEPS", "7"))

curiosity_scales = range(cs_min, stop=cs_max, length=cs_steps)
community_scales = range(cm_min, stop=cm_max, length=cm_steps)
rows = Vector{Dict{String,Any}}()

trials_per_cell = parse(Int, get(ENV, "DAY7_SWEEP_TRIALS_PER_CELL", "1000"))
@printf("Day-7 sweep: %d×%d grid, %d trials per cell -> ~%d total trials\n", length(curiosity_scales), length(community_scales), trials_per_cell, length(curiosity_scales)*length(community_scales)*trials_per_cell)

for cs in curiosity_scales
    for cm in community_scales
        ENV["DAY7_CURIOSITY_SCALE"] = string(cs)
        ENV["DAY7_COMMUNITY_SCALE"] = string(cm)
        res = run_day7("day6_mc_results.csv", trials=trials_per_cell, rng_seed=42)
        push!(rows, Dict("curiosity_scale"=>cs, "community_strength"=>cm, "trials"=>trials_per_cell, "rest_rate"=>res["rest_rate"], "cycle_rate"=>res["cycle_rate"], "mean_fidelity"=>res["mean_fidelity"]))
        for k in ("DAY7_CURIOSITY_SCALE","DAY7_COMMUNITY_SCALE")
            if haskey(ENV, k)
                delete!(ENV, k)
            end
        end
    end
end

out = joinpath(@__DIR__, "..", "sensitivity", "day7_rest_sweep.csv")
mkpath(joinpath(@__DIR__, "..", "sensitivity"))
CSV.write(out, DataFrame(rows))
println("Wrote sweep CSV: ", out)

# create heatmaps using Python helper
png_out = joinpath(@__DIR__, "..", "sensitivity", "day7_rest_sweep_heatmap.png")
try
    run(`python tools/plot_day7_sweep.py $(out) $(png_out)`)
    println("Wrote sweep heatmap: ", png_out)
catch err
    @warn "Failed to create Day-7 sweep plot" err
end
println("Day-7 sweep complete.")
