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
else
    dynamo_scales = [missing]
end
results = Vector{Dict{String,Any}}()
# default scaled sweep for tighter uncertainty bands; overridable with ENV["SWEEP_TRIALS"]
trials = parse(Int, get(ENV, "SWEEP_TRIALS", "10000"))  # set lower in tests via ENV when needed

for m in multipliers
    for ds in dynamo_scales
        tag = replace(string(m), '.'=>"p") * (ds === missing ? "" : "_d$(replace(string(ds), '.'=>"p"))")

        tmp = joinpath(@__DIR__, "..", "data", "planetary_mc_results_10k_enriched_scaled_$(tag).csv")
        tbl = deepcopy(pdf)
        has_hlab = any(x->x==:habitability_enriched || x=="habitability_enriched", names(tbl))
        if has_hlab
            colname = any(x->x==:habitability_enriched, names(tbl)) ? :habitability_enriched : "habitability_enriched"
            tbl[!, colname] = clamp.(Float64.(tbl[!, colname]) .* m, 0.0, 1.0)
        else
            error("habitability_enriched column missing in source CSV")
        end
        CSV.write(tmp, tbl)

        # Run Day-3 by invoking the existing runner with PLANET_CSV pointed at our scaled table
        println("Sweep m=$(m) ds=$(ds): running Day-3 MC (trials=$(trials)) using scaled prior -> $(tmp)")
        ENV["PLANET_CSV"] = tmp
        ENV["SWEEP_TRIALS"] = string(trials)
        ENV["DAY3_OUT"] = joinpath(@__DIR__, "..", "data", "day3_mc_results_gaia_scaled_$(tag).csv")
        if ds !== missing
            ENV["DYNAMO_SCALE"] = string(ds)
        end
        run(`julia --project=. tools/run_day3_mc_with_gaia_priors.jl`)
        # clear env keys to avoid cross-contamination
        for k in ["PLANET_CSV","SWEEP_TRIALS","DAY3_OUT","DYNAMO_SCALE"]
            if haskey(ENV, k)
                delete!(ENV, k)
            end
        end

        # chain Day-4, Day-5, Day-6 using the produced day3 output
        d3path = joinpath(@__DIR__, "..", "data", "day3_mc_results_gaia_scaled_$(tag).csv")
        println("Sweep m=$(m) ds=$(ds): running Day-4 MC (using $(d3path))")
        res4 = run_day4_mc(trials=trials, rng_seed=42, base_suction=3.5, suction_jitter=0.005, day3_path=d3path)
        println("Sweep m=$(m) ds=$(ds): running Day-5 MC")
        run_day5_mc(trials=trials, rng_seed=42)
        println("Sweep m=$(m) ds=$(ds): running Day-6 MC")
        res6 = run_day6_mc(trials=trials, rng_seed=42, day5_path="day5_mc_results.csv")

        push!(results, Dict("multiplier"=>m, "dynamo_scale"=>ds, "day3_success"=>missing, "day4_gov_rate"=>res4["gov_rate"], "day4_habitable_frac"=>res4["habitable_frac"], "day6_human_rate"=>res6["humanity_rate"], "mean_dominion"=>res6["mean_dominion"]))

        # chain Day-4, Day-5, Day-6 using the produced day3_mc_results_gaia.csv (baseline)
        println("Sweep m=$(m) ds=$(ds): running Day-4 MC (baseline day3) ")
        res4b = run_day4_mc(trials=trials, rng_seed=42, base_suction=3.5, suction_jitter=0.005, day3_path="day3_mc_results_gaia.csv")
        println("Sweep m=$(m) ds=$(ds): running Day-5 MC")
        run_day5_mc(trials=trials, rng_seed=42)
        println("Sweep m=$(m) ds=$(ds): running Day-6 MC")
        res6b = run_day6_mc(trials=trials, rng_seed=42, day5_path="day5_mc_results.csv")

        push!(results, Dict("multiplier"=>m, "dynamo_scale"=>ds, "day3_success"=>missing, "day4_gov_rate"=>res4b["gov_rate"], "day4_habitable_frac"=>res4b["habitable_frac"], "day6_human_rate"=>res6b["humanity_rate"], "mean_dominion"=>res6b["mean_dominion"]))
    end
end

out = joinpath(@__DIR__, "..", "sensitivity", "sensitivity_sweep_summary.csv")
mkpath(joinpath(@__DIR__, "..", "sensitivity"))
CSV.write(out, DataFrame(results))
println("Wrote sensitivity summary: ", out)
println("Sweep complete (quick mode).")
