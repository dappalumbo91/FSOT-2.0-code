using CSV
using DataFrames
using Downloads

"""host_from_pl_name(pl_name)
Extract host name from a planet name string (e.g. "Kepler-186 f" -> "Kepler-186").
"""
function host_from_pl_name(pl_name::AbstractString)
    # drop trailing component after last space
    if !occursin(' ', pl_name)
        return pl_name
    end
    host = join(split(pl_name)[1:end-1], " ")
    return strip(host)
end

"""resolve_coords_via_sesame(name)
Use CDS Sesame name resolver (HTTP) to obtain RA/DEC in degrees. Returns (ra, dec) or (NaN, NaN) on failure.
"""
function resolve_coords_via_sesame(name::AbstractString)
    url = "http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-o?name=$(replace(name, ' '=>'+'))"
    s = ""
    try
        @show url
        tmp = Downloads.download(url)
        @show tmp
        s = read(tmp, String)
        @show length(s)
    catch err
        @show err
        return (NaN, NaN)
    end
    # Sesame text output contains lines like "%J 19 12 12.3 +18 23 45"
    # Also may contain lines with 'coord: <ra> <dec>' or 'Right ascension' etc. We'll attempt to parse a line with 'J' or 'BJ' or 'coord'.
    for line in split(s, '\n')
        if startswith(strip(line), "%J")
            # format: %J <ra_deg> <dec_deg> or sexagesimal; sometimes already decimal
            parts = split(strip(line))
            if length(parts) >= 3
                try
                    ra = parse(Float64, parts[2])
                    dec = parse(Float64, parts[3])
                    return (ra, dec)
                catch
                    continue
                end
            end
        end
        if occursin("coord:", line)
            toks = split(line)
            for (i,t) in enumerate(toks)
                if t == "coord:" && i+2 <= length(toks)
                    try
                        ra = parse(Float64, toks[i+1]); dec = parse(Float64, toks[i+2]);
                        return (ra, dec)
                    catch
                    end
                end
            end
        end
    end
    return (NaN, NaN)
end

"""build_gaia_tap_query(ra, dec, radius_arcsec=2)
Return TAP sync URL for Gaia DR3 cone search around (ra,dec).
"""
function build_gaia_tap_query(ra::Real, dec::Real; radius_arcsec::Real=2.0)
    r_deg = radius_arcsec / 3600.0
    query = "SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, teff_gspphot, mh_gspphot FROM gaiadr3.gaia_source WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', $(ra), $(dec), $(r_deg)))=1 ORDER BY parallax DESC"
    url = "https://gea.esac.esa.int/tap-server/tap/sync?query=$(replace(query, ' '=>'+'))&format=csv"
    return url
end

"""query_gaia_by_coord(ra, dec; radius_arcsec=2)
Query Gaia TAP sync for nearest source near (ra, dec). Returns Dict or empty Dict on failure.
"""
function query_gaia_by_coord(ra::Real, dec::Real; radius_arcsec::Real=2.0)
    url = build_gaia_tap_query(ra, dec; radius_arcsec=radius_arcsec)
    try
        csvtext = String(download(url))
    catch err
        return Dict()
    end
    tbl = CSV.File(IOBuffer(csvtext)) |> DataFrame
    if nrow(tbl) == 0
        return Dict()
    end
    row = tbl[1,:]
    return Dict(
        :source_id => row.source_id,
        :ra => row.ra,
        :dec => row.dec,
        :parallax => row.parallax,
        :pmra => row.pmra,
        :pmdec => row.pmdec,
        :gmag => row.phot_g_mean_mag,
        :teff_gspphot => haskey(row, :teff_gspphot) ? row.teff_gspphot : missing,
        :mh_gspphot => haskey(row, :mh_gspphot) ? row.mh_gspphot : missing
    )
end

"""crossmatch_archive_with_gaia(archive_csv, out_gaia_csv; limit=500)
Cross‑match unique host names from an Exoplanet Archive CSV with Gaia (SESAME → Gaia TAP).
Writes a CSV of enriched host rows with Gaia fields. limit controls number of unique hosts to query (default 500).
"""
function crossmatch_archive_with_gaia(archive_csv::AbstractString, out_gaia_csv::AbstractString; limit::Int=500)
    println("crossmatch: loading archive CSV at: ", archive_csv)
    rows = CSV.File(archive_csv) |> DataFrame
    println("crossmatch: columns -> ", names(rows))
    has_pl_name = any(x->(x==:pl_name) || (x=="pl_name"), names(rows))
    if !has_pl_name
        error("Input archive CSV must contain pl_name column")
    end
    # access pl_name robustly whether column name is Symbol or String
    plcol = any(x->x==:pl_name, names(rows)) ? :pl_name : "pl_name"
    hosts = unique([host_from_pl_name(String(p)) for p in rows[!, plcol]])
    hosts = hosts[1:min(limit, length(hosts))]
    results = Vector{Dict{String,Any}}()
    # create a lookup for rows by hostname if ra/dec present
    has_coords = any(x->(x==:ra)||(x=="ra"), names(rows)) && any(x->(x==:dec)||(x=="dec"), names(rows))
    host_to_coord = Dict{String,Tuple{Float64,Float64}}()
    if has_coords && (:hostname in names(rows) || "hostname" in names(rows))
        hostcol = (:hostname in names(rows)) ? :hostname : "hostname"
        for r in eachrow(rows)
            hn = String(r[hostcol])
            if !ismissing(r[:ra]) && !ismissing(r[:dec])
                try
                    host_to_coord[hn] = (Float64(r[:ra]), Float64(r[:dec]))
                catch
                end
            end
        end
    end

    for h in hosts
        ra = NaN; dec = NaN
        if haskey(host_to_coord, h)
            ra, dec = host_to_coord[h]
        else
            ra, dec = resolve_coords_via_sesame(h)
        end

        if isnan(ra) || isnan(dec)
            push!(results, Dict("host"=>h, "ra"=>NaN, "dec"=>NaN, "gaia_source_id"=>missing, "parallax"=>missing, "pmra"=>missing, "pmdec"=>missing, "teff_gspphot"=>missing, "mh_gspphot"=>missing))
            continue
        end
        g = query_gaia_by_coord(ra, dec; radius_arcsec=2.0)
        if isempty(g)
            push!(results, Dict("host"=>h, "ra"=>ra, "dec"=>dec, "gaia_source_id"=>missing, "parallax"=>missing, "pmra"=>missing, "pmdec"=>missing, "teff_gspphot"=>missing, "mh_gspphot"=>missing))
        else
            push!(results, Dict("host"=>h, "ra"=>g[:ra], "dec"=>g[:dec], "gaia_source_id"=>g[:source_id], "parallax"=>g[:parallax], "pmra"=>g[:pmra], "pmdec"=>g[:pmdec], "teff_gspphot"=>g[:teff_gspphot], "mh_gspphot"=>g[:mh_gspphot]))
        end
    end
    df = DataFrame(results)
    CSV.write(out_gaia_csv, df)
    return out_gaia_csv
end

"""apply_gaia_enrichment(mc_csv, gaia_csv, out_csv)
Join MC results with Gaia host enrichments by derived host name and apply simple bonuses:
- +0.05 habitability if host has mh_gspphot > 0
- +0.03 habitability if teff_gspphot present and in 3000..7000
Returns path to enriched CSV.
"""
function apply_gaia_enrichment(mc_csv::AbstractString, gaia_csv::AbstractString, out_csv::AbstractString)
    mc = CSV.File(mc_csv) |> DataFrame
    gaia = CSV.File(gaia_csv) |> DataFrame
    # add host column to mc by parsing pl_name
    mc[!, :host] = [host_from_pl_name(String(p)) for p in mc.pl_name]
    # ensure gaia has host column (accept string or symbol)
    hostcol = any(x->x==:host, names(gaia)) ? :host : (any(x->x=="host", names(gaia)) ? "host" : nothing)
    if hostcol === nothing
        error("Gaia table missing host column")
    end
    gmap = Dict(String(row[hostcol]) => row for row in eachrow(gaia))
    enriched_scores = Float64[]
    for row in eachrow(mc)
        base = try parse(Float64, string(row.habitability_score)) catch; NaN end
        bonus = 0.0
        h = row.host
        if haskey(gmap, h)
            grec = gmap[h]
            if haskey(grec, :mh_gspphot) && !ismissing(grec.mh_gspphot)
                try
                    mh = parse(Float64, string(grec.mh_gspphot))
                    if mh > 0
                        bonus += 0.05
                    end
                catch
                end
            end
            if haskey(grec, :teff_gspphot) && !ismissing(grec.teff_gspphot)
                try
                    tt = parse(Float64, string(grec.teff_gspphot))
                    if tt > 3000 && tt < 7000
                        bonus += 0.03
                    end
                catch
                end
            end
        end
        newscore = clamp(isnan(base) ? 0.0 : base + bonus, 0.0, 1.0)
        push!(enriched_scores, newscore)
    end
    mc[!, :habitability_enriched] = enriched_scores
    CSV.write(out_csv, mc)
    return out_csv
end
