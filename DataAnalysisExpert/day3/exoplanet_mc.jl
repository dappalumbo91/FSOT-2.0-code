using CSV
using DataFrames
using Random
include("planetary_db.jl")

"""habitability_score(classification)
Compute a 0..1 habitability score from classification Dict produced by classify_planet.
Weights are tunable and intentionally simple for MC scaffolding.
"""
function habitability_score(cls::Dict{Symbol,T}) where {T}
    score = 0.0
    score += cls[:is_rocky] ? 0.35 : 0.0
    score += cls[:low_e] ? 0.20 : 0.0
    score += 0.35 * cls[:em_retention]
    # small bonus for temperate host proxy (teff ~ 4000..6500)
    teff = cls[:teff]
    if !isnan(teff) && teff > 3000 && teff < 7000
        score += 0.10
    end
    return clamp(score, 0.0, 1.0)
end

"""run_exoplanet_mc(in_csv, out_csv; n=100)
Read planets from CSV, compute classifications + habitability_score, write out results CSV.
"""
function run_exoplanet_mc(in_csv::AbstractString, out_csv::AbstractString; n::Int=100)
    rows = load_exoplanet_csv(in_csv)
    results = Vector{Dict{String,Any}}()
    for r in rows
        cls = classify_planet(r)
        if haskey(r, "st_teff") && !ismissing(r["st_teff"])
            try
                cls[:teff] = Float64(r["st_teff"])
            catch
                cls[:teff] = NaN
            end
        else
            cls[:teff] = NaN
        end
        score = habitability_score(cls)
        # add coarse bucket
        bucket = bucket_classification(r)
        push!(results, merge(r, Dict("habitability_score"=>score, "class"=>String(bucket))))
    end
    # write CSV
    df = DataFrame(results)
    CSV.write(out_csv, df)
    return out_csv
end

"""run_resample_mc(in_csv, out_csv; n)
Resample rows with replacement to produce n rows (useful to reach 10k trials).
This simply replicates rows and preserves classification/habitability scoring.
"""

function run_resample_mc(in_csv::AbstractString, out_csv::AbstractString; n::Int=10000)
    base_rows = load_exoplanet_csv(in_csv)
    if isempty(base_rows)
        error("Input CSV has no rows")
    end
    N = length(base_rows)
    rng = Random.MersenneTwister(12345)
    results = Vector{Dict{String,Any}}()
    for i in 1:n
        idx = rand(rng, 1:N)
        r = deepcopy(base_rows[idx])
        # optionally jitter numeric fields slightly (small relative noise)
        for key in ("pl_rade","pl_bmasse","pl_orbeccen","pl_insol")
            if haskey(r, key) && !ismissing(r[key])
                try
                    v = parse(Float64, string(r[key]))
                    if !isnan(v) && isfinite(v)
                        noise = 1.0 + 0.02*(rand(rng)-0.5)
                        r[key] = v*noise
                    end
                catch
                end
            end
        end
        push!(results, r)
    end
    # run MC scoring on the resampled table
    tmp_in = out_csv * ".resample_in.csv"
    df = DataFrame(results)
    CSV.write(tmp_in, df)
    res = run_exoplanet_mc(tmp_in, out_csv; n=n)
    rm(tmp_in; force=true)
    return res
end

"""summarize_mc_results(in_csv, summary_csv; topn=10)
Produce classification breakdown, low-e fraction, and top-N habitability table (CSV).
Returns the summary dict.
"""
function summarize_mc_results(in_csv::AbstractString, summary_csv::AbstractString; topn::Int=10)
    tbl = CSV.File(in_csv) |> DataFrame
    total = nrow(tbl)
    classes = combine(groupby(tbl, :class), nrow => :count)
    lowe_frac = count(x->(haskey(x, :pl_orbeccen) && !ismissing(x[:pl_orbeccen]) && tryparse(Float64, string(x[:pl_orbeccen])) !== nothing ? parse(Float64, string(x[:pl_orbeccen])) : NaN) < 0.1, eachrow(tbl)) / total

    # safe habitability column (accept Symbol or String column names)
    colname = if any(x->x==:habitability_score, names(tbl))
        :habitability_score
    elseif any(x->x=="habitability_score", names(tbl))
        "habitability_score"
    else
        error("Input CSV missing habitability_score")
    end

    top = sort(tbl, [colname], rev=true)[1:min(topn, total), :]

    # write summary CSVs
    CSV.write(summary_csv * ".classes.csv", classes)
    CSV.write(summary_csv * ".top.csv", top)

    # produce a summary Dict to return
    summary = Dict(
        :total => total,
        :class_counts => Dict(classes.class .=> classes.count),
        :low_e_fraction => lowe_frac,
        :top_n => topn
    )
    return summary
end

"""run_exoplanet_archive_mc(out_csv; download_path, n)
Convenience wrapper: downloads the Exoplanet Archive CSV (if download_path provided) and runs the MC.
If network/download fails, throws an error.
"""
function run_exoplanet_archive_mc(out_csv::AbstractString; download_path::AbstractString="data/exoplanet_archive.csv", n::Int=10000)
    # attempt to download full catalog
    try
        fetch_exoplanet_archive_csv(download_path)
    catch e
        error("Exoplanet Archive download failed: ", e)
    end
    return run_exoplanet_mc(download_path, out_csv; n=n)
end
