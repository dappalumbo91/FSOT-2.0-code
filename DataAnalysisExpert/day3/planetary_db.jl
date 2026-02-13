using CSV
using DataFrames
@eval using Downloads; true

# Simple planetary DB loader and classifiers for FSOT

"""load_exoplanet_csv(path)
Read a CSV (NASA Exoplanet Archive style) and return a Vector of Dict rows.
"""
function load_exoplanet_csv(path::AbstractString)
    df = CSV.File(path) |> DataFrame
    rows = Vector{Dict{String,Any}}()
    for r in eachrow(df)
        push!(rows, Dict(string(k)=>r[k] for k in names(df)))
    end
    return rows
end

"""classify_planet(row)
Return a small classification Dict for simple FSOT scoring.
Fields: :is_rocky, :low_e, :em_retention (0..1)
"""
function classify_planet(row::Dict{String,Any})
    # safe access helpers
    getf(k, default) = haskey(row, k) && !ismissing(row[k]) ? row[k] : default

    r = try Float64(getf("pl_rade", NaN)) catch; NaN end
    m = try Float64(getf("pl_bmasse", NaN)) catch; NaN end
    e = try Float64(getf("pl_orbeccen", NaN)) catch; NaN end
    insol = try Float64(getf("pl_insol", NaN)) catch; NaN end
    teff = try Float64(getf("st_teff", NaN)) catch; NaN end

    is_rocky = !isnan(r) && r <= 2.0
    low_e = !isnan(e) && e < 0.1

    # crude EM retention proxy: bigger mass + lower insolation -> higher retention
    mass_term = isnan(m) ? 0.5 : clamp((log10(m+1) / 2.0), 0.0, 1.0)
    insol_term = isnan(insol) ? 0.5 : clamp(1.0 - (insol / 10.0), 0.0, 1.0)
    em_retention = clamp(0.2*mass_term + 0.8*insol_term, 0.0, 1.0)

    return Dict(
        :is_rocky => is_rocky,
        :low_e => low_e,
        :em_retention => em_retention,
        :radius => r,
        :mass => m,
        :ecc => e,
        :insol => insol,
        :teff => teff
    )
end

"""bucket_classification(row)
Return a coarse class string: :rocky, :mini_neptune, :gas_giant, :rogue, :hot_jupiter, :eccentric
"""
function bucket_classification(row::Dict{String,Any})
    r = try Float64(haskey(row,"pl_rade") ? row["pl_rade"] : NaN) catch; NaN end
    m = try Float64(haskey(row,"pl_bmasse") ? row["pl_bmasse"] : NaN) catch; NaN end
    e = try Float64(haskey(row,"pl_orbeccen") ? row["pl_orbeccen"] : NaN) catch; NaN end

    if isnan(r) && isnan(m)
        return :unknown
    end
    # rogue — no host TEFF or insolation ~0
    insol = try Float64(haskey(row,"pl_insol") ? row["pl_insol"] : NaN) catch; NaN end
    if !isnan(insol) && insol == 0.0
        return :rogue
    end

    if !isnan(r) && r <= 2.0
        return :rocky
    elseif !isnan(r) && r > 2.0 && r <= 6.0
        return :mini_neptune
    elseif !isnan(r) && r > 6.0
        # hot Jupiter-like if very close (a small sma or high insolation)
        if haskey(row, "pl_orbsmax") && !ismissing(row["pl_orbsmax"]) && !isnan(row["pl_orbsmax"]) && Float64(row["pl_orbsmax"]) < 0.1
            return :hot_jupiter
        end
        return :gas_giant
    end
    # fallback to mass heuristic
    if !isnan(m) && m > 50
        return :gas_giant
    elseif !isnan(m) && m > 5
        return :mini_neptune
    end
    return :unknown
end

"""sample_planetary_stats(rows)
Return mean and min/max for key properties.
"""
function sample_planetary_stats(rows::Vector{Dict{String,Any}})
    radii = Float64[]
    eccs = Float64[]
    for r in rows
        if haskey(r, "pl_rade") && !ismissing(r["pl_rade"]) && !isnan(r["pl_rade"])
            push!(radii, Float64(r["pl_rade"]))
        end
        if haskey(r, "pl_orbeccen") && !ismissing(r["pl_orbeccen"]) && !isnan(r["pl_orbeccen"])
            push!(eccs, Float64(r["pl_orbeccen"]))
        end
    end
    return Dict(
        :r_mean => isempty(radii) ? NaN : mean(radii),
        :r_min => isempty(radii) ? NaN : minimum(radii),
        :r_max => isempty(radii) ? NaN : maximum(radii),
        :e_mean => isempty(eccs) ? NaN : mean(eccs),
        :e_min => isempty(eccs) ? NaN : minimum(eccs),
        :e_max => isempty(eccs) ? NaN : maximum(eccs)
    )
end

# -----------------------
# Earth baseline & planetary interior priors
# -----------------------

"""get_earth_baseline()
Return a Dict of Earth-like baseline priors used to seed tectonic/ocean/weather MC priors.
Fields: :tectonic_rate (cm/yr), :volcanic_index (0..1), :ocean_fraction (0..1), :dynamo_scale (0..2), :core_heat_TW
"""
function get_earth_baseline()
    return Dict(
        :tectonic_rate => 2.5,           # cm/yr (order-of-magnitude Earth plate rates)
        :volcanic_index => 0.85,         # proxy for active volcanism/outgassing
        :ocean_fraction => 0.71,         # Earth ocean coverage fraction
        :dynamo_scale => 1.0,            # Earth-normal dynamo strength
        :core_heat_TW => 47.0            # Earth internal heat power (TW, order estimate)
    )
end

# -----------------------
# NASA Exoplanet Archive ingestion helpers
# -----------------------

"""build_exoplanet_archive_query(cols)
Return a TAP sync query URL (CSV) for the `ps` table with selected columns.
"""
function build_exoplanet_archive_query(cols::Vector{String}=String[
        "pl_name","hostname","pl_rade","pl_bmasse","pl_orbeccen","pl_orbsmax",
        "ra","dec","st_teff","st_mass","st_rad","pl_insol"])
    select_clause = join(cols, ",")
    q = "select+$(replace(select_clause,' '=>'+'))+from+ps"
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=$(q)&format=csv"
    return url
end

"""fetch_exoplanet_archive_csv(outpath; cols)
Download the Exoplanet Archive `ps` table (CSV) to outpath. Uses lazy import of HTTP.
Note: network required and `HTTP` Julia package must be installed.
"""
function fetch_exoplanet_archive_csv(outpath::AbstractString; cols=String[])
    # use Downloads (stdlib) to avoid third-party dependency
    url = isempty(cols) ? build_exoplanet_archive_query() : build_exoplanet_archive_query(cols)
    try
        Downloads.download(url, outpath)
    catch err
        error("Failed to download Exoplanet Archive CSV: ", err)
    end
    return outpath
end

# --- Public data ingestion helpers (USGS / NOAA / PREM) ---
function fetch_usgs_earthquakes(outpath::AbstractString; starttime::String="2024-01-01", endtime::String=Dates.format(Dates.today(), "yyyy-mm-dd"), minmagnitude::Float64=4.5)
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?starttime=$(starttime)&endtime=$(endtime)&minmagnitude=$(minmagnitude)&limit=2000"
    try
        Downloads.download(url, outpath)
    catch err
        error("Failed to download USGS earthquakes: ", err)
    end
    return outpath
end

function summarize_usgs_earthquakes(geojson_path::AbstractString)
    if !isfile(geojson_path)
        error("USGS GeoJSON not found: " * geojson_path)
    end
    txt = read(geojson_path, String)
    # simple heuristic parsing for count and mean depth/mag
    mags = Float64[]
    depths = Float64[]
    for m in eachmatch(r"\"mag\":\s*([0-9\.eE\-]+)", txt)
        push!(mags, parse(Float64, m.captures[1]))
    end
    for d in eachmatch(r"\"depth\":\s*([0-9\.eE\-]+)", txt)
        push!(depths, parse(Float64, d.captures[1]))
    end
    return Dict(:count=>length(mags),
                :mean_mag => (isempty(mags) ? NaN : mean(mags)),
                :mean_depth_km => (isempty(depths) ? NaN : mean(depths)))
end

function fetch_prem_table(outpath::AbstractString)
    # lightweight PREM excerpt (use a hosted CSV if available). If not, write a small built-in sample
    sample = "radius_km,density,vs,vp\n0,13.0,0,11.0\n1221,12.1,3.5,10.5\n3480,9.9,3.6,10.2\n6371,2.6,3.0,5.0\n"
    open(outpath, "w") do io
        write(io, sample)
    end
    return outpath
end

# --- Paleobiology / hominid fetch + simple summarizer ---
function fetch_paleobio_hominins(outpath::AbstractString; genus::AbstractString="Homo", limit::Int=500)
    # uses the Paleobiology Database occurrences API (public)
    # example: https://paleobiodb.org/data1.2/occs/list.json?base_name=Homo&limit=500
    url = "https://paleobiodb.org/data1.2/occs/list.json?base_name=$(genus)&limit=$(limit)&show=ident"
    try
        Downloads.download(url, outpath)
    catch err
        error("Failed to download Paleobiology DB occurrences: ", err)
    end
    return outpath
end

function summarize_paleobio_ages(json_path::AbstractString)
    if !isfile(json_path)
        error("Paleobiology JSON not found: " * json_path)
    end
    txt = read(json_path, String)
    ages = Float64[]
    # extract any numeric age fields commonly used by the API: 'max_ma', 'min_ma', 'early_age', 'late_age'
    for m in eachmatch(r"\"max_ma\"\s*:\s*([0-9\.eE\-]+)", txt)
        push!(ages, parse(Float64, m.captures[1]))
    end
    for m in eachmatch(r"\"min_ma\"\s*:\s*([0-9\.eE\-]+)", txt)
        push!(ages, parse(Float64, m.captures[1]))
    end
    for m in eachmatch(r"\"early_age\"\s*:\s*([0-9\.eE\-]+)", txt)
        push!(ages, parse(Float64, m.captures[1]))
    end
    for m in eachmatch(r"\"late_age\"\s*:\s*([0-9\.eE\-]+)", txt)
        push!(ages, parse(Float64, m.captures[1]))
    end

    if isempty(ages)
        return Dict(:count=>0, :mean=>NaN, :std=>NaN, :n=>0)
    end
    return Dict(:count=>length(ages), :mean=>mean(ages), :std=>std(ages), :n=>length(ages))
end

"""map_exoplanet_archive_row(row)
Map known Exoplanet Archive columns to the internal row Dict format (safe typing).
"""
function map_exoplanet_archive_row(row::Dict{String,Any})
    # ensure expected keys exist (coerce missing to NaN)
    mapped = Dict{String,Any}()
    for k in ("pl_name","pl_rade","pl_bmasse","pl_orbeccen","pl_orbsmax","st_teff","st_mass","st_rad","pl_insol")
        mapped[k] = haskey(row, k) ? row[k] : missing
    end
    return mapped
end

