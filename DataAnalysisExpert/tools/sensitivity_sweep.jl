println("Starting sensitivity sweep for Gaia enrichment (quick mode: trials=2000)")
include(joinpath(@__DIR__, "..", "day3_mc.jl"))
include(joinpath(@__DIR__, "..", "day4_mc.jl"))
include(joinpath(@__DIR__, "..", "day5_mc.jl"))
include(joinpath(@__DIR__, "..", "day6_mc.jl"))
using CSV, DataFrames

src = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k_enriched.csv")
if !isfile(src)
    error("Enriched planetary CSV not found: ", src)
end
pdf = CSV.File(src) |> DataFrame

multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
# optional extended axis: metallicity -> dynamo mapping
extended = parse(Bool, get(ENV, "SWEEP_EXTENDED", "false"))
if extended
    dynamo_scales = [0.8, 1.0, 1.2]
    tectonic_scales = [0.5, 1.0, 1.5]      # relative tectonic activity multiplier
    ocean_fracs = [0.5, 0.71, 0.9]
    weather_idxs = [0.5, 1.0, 1.5]
    # when extended, optionally sample a subset of the full grid (useful for quick runs)
    sample_size = parse(Int, get(ENV, "SWEEP_SAMPLE", "0"))
else
    dynamo_scales = [missing]
    tectonic_scales = [missing]
    ocean_fracs = [missing]
    weather_idxs = [missing]
    sample_size = 0
end
results = Vector{Dict{String,Any}}()
# default scaled sweep for tighter uncertainty bands; overridable with ENV["SWEEP_TRIALS"]
trials = parse(Int, get(ENV, "SWEEP_TRIALS", "10000"))  # set lower in tests via ENV when needed

# build list of parameter combinations (support sampling if enabled)
param_list = []
if sample_size > 0 && extended
    # random sample from full grid for quick exploration
    full_grid = [(m, ds, tr, of, wi) for m in multipliers for ds in dynamo_scales for tr in tectonic_scales for of in ocean_fracs for wi in weather_idxs]
    rngs = MersenneTwister(42)
    idxs = rand(rngs, 1:length(full_grid), min(sample_size, length(full_grid)))
    for i in idxs
        push!(param_list, full_grid[i])
    end
else
    for m in multipliers
        for ds in dynamo_scales
            for tr in tectonic_scales
                for of in ocean_fracs
                    for wi in weather_idxs
                        push!(param_list, (m, ds, tr, of, wi))
                    end
                end
            end
        end
    end
end

for (m, ds, tr, of, wi) in param_list
    tag = replace(string(m), '.'=>"p") * (ds === missing ? "" : "_d$(replace(string(ds), '.'=>"p"))") * (tr === missing ? "" : "_t$(replace(string(tr), '.'=>"p"))") * (of === missing ? "" : "_o$(replace(string(of), '.'=>"p"))") * (wi === missing ? "" : "_w$(replace(string(wi), '.'=>"p"))")

    tmp = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k_enriched_scaled_$(tag).csv")
    tbl = deepcopy(pdf)
    has_hlab = any(x->x==:habitability_enriched || x=="habitability_enriched", names(tbl))
    if has_hlab
        colname = any(x->x==:habitability_enriched, names(tbl)) ? :habitability_enriched : "habitability_enriched"
        tbl[!, colname] = clamp.(Float64.(tbl[!, colname]) .* m, 0.0, 1.0)
        # inject tectonic/ocean/weather priors (sampled scalars applied as modifications)
        tbl[!, :tectonic_rate] = tr === missing ? fill(NaN, nrow(tbl)) : fill(tr, nrow(tbl))
        tbl[!, :ocean_fraction] = of === missing ? fill(NaN, nrow(tbl)) : fill(of, nrow(tbl))
        tbl[!, :weather_index] = wi === missing ? fill(NaN, nrow(tbl)) : fill(wi, nrow(tbl))
    else
        error("habitability_enriched column missing in source CSV")
    end
    CSV.write(tmp, tbl)

    # Run Day-3 by invoking the existing runner with PLANET_CSV pointed at our scaled table
    println("Sweep m=$(m) ds=$(ds) tr=$(tr) of=$(of) wi=$(wi): running Day-3 MC -> $(tmp)")
    ENV["PLANET_CSV"] = tmp
    ENV["SWEEP_TRIALS"] = string(trials)
    ENV["DAY3_OUT"] = joinpath(@__DIR__, "..", "data", "day3_mc_results_gaia_scaled_$(tag).csv")
    if ds !== missing
        ENV["DYNAMO_SCALE"] = string(ds)
    end
    # propagate tectonic/ocean/weather as environment overrides for Day-3 runner
    if tr !== missing
        ENV["SAMP_TECTONIC_RATE"] = string(tr)
    end
    if of !== missing
        ENV["SAMP_OCEAN_FRACTION"] = string(of)
    end
    if wi !== missing
        ENV["SAMP_WEATHER_INDEX"] = string(wi)
    end

    run(`julia --project=. tools/run_day3_mc_with_gaia_priors.jl`)
    # clear env keys to avoid cross-contamination
    for k in ("PLANET_CSV","SWEEP_TRIALS","DAY3_OUT","DYNAMO_SCALE","SAMP_TECTONIC_RATE","SAMP_OCEAN_FRACTION","SAMP_WEATHER_INDEX")
        if haskey(ENV, k)
            delete!(ENV, k)
        end
    end

    # chain Day-4, Day-5, Day-6 using the produced day3 output
    d3path = joinpath(@__DIR__, "..", "data", "day3_mc_results_gaia_scaled_$(tag).csv")
    println("Sweep params $(tag): running Day-4 MC (using $(d3path))")
    res4 = run_day4_mc(trials=trials, rng_seed=42, base_suction=3.5, suction_jitter=0.005, day3_path=d3path)
    println("Sweep params $(tag): running Day-5 MC")
    res5 = run_day5_mc(trials=trials, rng_seed=42)

    push!(results, Dict("multiplier"=>m, "dynamo_scale"=>ds, "tectonic_mult"=>tr, "ocean_frac"=>of, "weather_idx"=>wi, "day4_hab_rate"=>res4["habitable_frac"], "day5_swarm"=>res5["swarming_rate"], "day5_cambrian"=>res5["cambrian_rate"], "mean_dominion"=>NaN))
end

out = joinpath(@__DIR__, "..", "sensitivity", "sensitivity_sweep_summary.csv")
mkpath(joinpath(@__DIR__, "..", "sensitivity"))
CSV.write(out, DataFrame(results))
println("Wrote sensitivity summary: ", out)

# produce 3D scatter visualization (tectonic, ocean, weather)
png_out = joinpath(@__DIR__, "..", "sensitivity", "sensitivity_3d.png")
try
    run(`python tools/plot_sensitivity_3d.py $(out) $(png_out)`)
    println("Wrote 3D sensitivity plot: ", png_out)
catch err
    @warn "Failed to create 3D plot" err
end

println("Sweep complete (quick mode).")
