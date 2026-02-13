# Day 1 — FSOT 2.0: Light Emergence (Julia)
# Purpose: compute FSOT Day-1 scalar, anchor to observed CMB constants,
# optionally fetch FIRAS monopole spectrum, save outputs and simple visualization.

using Printf
# JSON may not be installed in the local Julia environment used for tests; use a fallback parser when needed
has_json = try
    @eval using JSON; true
catch
    false
end
using LinearAlgebra

# Optional data download / plotting
has_downloads = try
    @eval using Downloads; true
catch
    false
end
# Defer plotting backend initialization to runtime to avoid heavy package precompile during tests
# Set to true only if you have Plots.jl available and want visualization.
has_plots = false

# --- Anchored observational constants (Planck / FIRAS) ---
const REAL_CMB_TEMP = BigFloat("2.72548")           # K (Fixsen / Planck)
const REAL_RHO_R_TODAY = BigFloat("4.64e-34")      # J/m^3 (radiation density proxy)
const Z_RECOMB = BigFloat(1090)                     # Approx recombination redshift

# --- High-precision FSOT-like constants (Day-1 tuning) ---
setprecision(128)
const φ = big((1 + sqrt(big(5))) / 2)
const ψ_con = BigFloat("0.288788095086602421278899932936033837486")
const POOF_FACTOR = BigFloat("1.618")
const SUCTION_FACTOR = BigFloat("1.618")

# Compute S for Day 1 using a simplified FSOT formula tuned for emergence
function compute_S_day1(; D_eff::BigFloat = BigFloat(25), Δψ::BigFloat = φ + BigFloat("0.618"),
                        phase_variance::BigFloat = BigFloat("0.01"), recent_hits::Int = 0,
                        observed::Bool = false)

    η_eff = φ * (1 + phase_variance)
    θ_s = pi / BigFloat(4)

    # Terms (kept deliberately simple & interpretable)
    poof_term = POOF_FACTOR * cos((ψ_con + Δψ) / η_eff) * exp(Δψ / 3)
    suction_term = -SUCTION_FACTOR * sin(θ_s) * ((D_eff - 25) / 25 + 1)
    growth_term = exp(BigFloat(1)) * (log(D_eff + 1) + (observed ? ψ_con : BigFloat(0)))
    quirk_term = (observed ? exp(ψ_con * phase_variance) * cos(Δψ) : cos(Δψ) * BigFloat(0.618))

    S = poof_term + suction_term + growth_term + quirk_term
    return S
end

# Anchor raw S to an "anchored radiation density" using observed constants
function anchor_to_cmb(S::BigFloat)
    emergence = S > 0
    light_density_raw = emergence ? exp(S) : BigFloat(0)

    early_scale = (1 + Z_RECOMB)^4
    anchored_density = light_density_raw * REAL_RHO_R_TODAY * early_scale

    return (emergence=emergence, light_density_raw=light_density_raw, anchored_density=anchored_density)
end

# Optional: attempt to download FIRAS monopole spectrum (if Downloads.jl available)
function try_fetch_firas(local_path::String = "firas_monopole_spec.txt")
    if !has_downloads
        return (ok=false, msg="Downloads.jl not available")
    end

    # LAMBDA paths can change; try a likely location and fall back to known mirror if necessary
    urls = [
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spectrum/firas_monopole_spec_v1.txt",
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spectrum/firas_monopole_spec.txt"
    ]
    for u in urls
        try
            Downloads.download(u, local_path)
            return (ok=true, path=local_path, url=u)
        catch e
            # continue trying other URLs
        end
    end
    return (ok=false, msg="All fetch attempts failed; network or URL changed")
end

# --- Planck / FIRAS local parsing helpers ---
# parse_firas_file: reads a simple whitespace-delimited FIRAS-style text file and
# returns an array of Float64 rows (skips comments and headers).
function parse_firas_file(path::String)
    lines = readlines(path)
    data = Array{Vector{Float64}}(undef, 0)
    for line in lines
        s = strip(line)
        if isempty(s) || startswith(s, "#")
            continue
        end
        parts = split(s)
        # skip header rows where any part is not numeric (handles scientific notation correctly)
        if any(p -> tryparse(Float64, p) === nothing, parts)
            continue
        end
        push!(data, parse.(Float64, parts))
    end
    return data
end

# parse_planck_json: simple JSON parser for Planck-like parameter outputs
function parse_planck_json(path::String)
    content = read(path, String)
    if has_json
        js = JSON.parse(content)
        T = haskey(js, "T_CMB") ? BigFloat(js["T_CMB"]) : REAL_CMB_TEMP
        z = haskey(js, "z_recomb") ? BigFloat(js["z_recomb"]) : Z_RECOMB
        return (T_CMB = T, z_recomb = z)
    else
        # Simple regex-based fallback for small JSON snippets
        mT = match(r"\"T_CMB\"\s*:\s*([0-9eE+\-.]+)", content)
        mz = match(r"\"z_recomb\"\s*:\s*([0-9eE+\-.]+)", content)
        T = mT !== nothing ? BigFloat(mT.captures[1]) : REAL_CMB_TEMP
        z = mz !== nothing ? BigFloat(mz.captures[1]) : Z_RECOMB
        return (T_CMB = T, z_recomb = z)
    end
end

# Convenience: read observed anchors from local Planck file and override constants (best-effort)
function load_anchors_from_planck(path::String)
    params = parse_planck_json(path)
    return (T_CMB = params.T_CMB, rho_r_today = REAL_RHO_R_TODAY, z_recomb = params.z_recomb)
end

# --- Additional parsers: FITS & LAMBDA table formats ---
# parse_lambda_table: handles common LAMBDA/Planck whitespace-delimited tables
# (skips comment lines starting with #, returns Vector{Vector{Float64}})
function parse_lambda_table(path::String)
    lines = readlines(path)
    rows = Vector{Vector{Float64}}()
    for line in lines
        s = strip(line)
        isempty(s) && continue
        startswith(s, "#") && continue
        parts = split(s)
        # try parsing numeric columns; skip header rows
        if any(p -> tryparse(Float64, p) === nothing, parts)
            continue
        end
        push!(rows, parse.(Float64, parts))
    end
    return rows
end

# parse_fits_file: try to read FITS using FITSIO.jl if available; otherwise accept
# a plain-text fallback (useful for test fixtures or converted files).
has_fitsio = try
    @eval using FITSIO; true
catch
    false
end

# Read FITS ASCII header blocks (first 2880-byte records) and return header Dict
function parse_fits_header(path::String)
    kv = Dict{String,String}()
    try
        open(path, "r") do io
            # read in 2880-byte blocks until END keyword found (read as bytes, safe for binary)
            header_str = ""
            while true
                bytes = read(io, 2880)
                # convert bytes to String (FITS header is ASCII -> valid UTF-8)
                block = String(bytes)
                header_str *= block
                if occursin("END", block)
                    break
                end
                if eof(io)
                    break
                end
            end
            # parse 80-char card images
            n = length(header_str)
            i = 1
            while i + 79 <= n
                card = header_str[i:i+79]
                key = strip(card[1:8])
                if !isempty(key)
                    if occursin('=', card)
                        valpart = strip(card[11:80])
                        val = split(valpart, '/')[1]
                        kv[key] = strip(val)
                    else
                        kv[key] = ""
                    end
                    if key == "END"
                        break
                    end
                end
                i += 80
            end
        end
    catch e
        return Dict(:ok => false, :msg => string(e))
    end
    return kv
end

function parse_fits_file(path::String)
    # first try header read (pure-Julia, lightweight)
    header = parse_fits_header(path)

    # If header parsing failed, return that failure immediately
    if isa(header, Dict) && haskey(header, :ok) && header[:ok] == false
        return Dict(:ok => false, :msg => header[:msg])
    end

    if has_fitsio
        try
            f = FITSIO.FITS(path)
            data = read(f[1])
            close(f)
            # normalize into Vector{Vector{Float64}} rows if possible
            rows = nothing
            if ndims(data) == 1
                rows = [Float64.(data)]
            elseif ndims(data) == 2
                rows = [Float64.(data[i, :]) for i in 1:size(data,1)]
            else
                rows = [Float64.(vec(data))]
            end
            return Dict(:ok => true, :header => header, :data => rows)
        catch e
            # fall back to header-only result below
        end
    end

    # If header indicates a FITS binary (SIMPLE present) but FITSIO not available,
    # return header and a message explaining FITSIO.jl is required to read data.
    if isa(header, Dict) && haskey(header, "SIMPLE")
        return Dict(:ok => false, :header => header, :msg => "FITSIO.jl not installed; header parsed only")
    end

    # Otherwise treat as whitespace table (useful for ASCII-table fixtures)
    rows = parse_lambda_table(path)
    return Dict(:ok => true, :header => header, :data => rows)
end

# Simple plotting helper (if Plots available)
function plot_dummy_wavefront(outpath::String = "day1_wavefront.png")
    if !has_plots
        return (ok=false, msg="Plots.jl not installed")
    end
    grid_size = 300
    time_steps = 150
    light_field = zeros(Float64, grid_size)
    light_field[grid_size ÷ 2] = 1.0
    wavefront = Float64[]
    for t in 1:time_steps
        light_field[2:end] .+= 0.06 * diff(light_field)
        light_field[1] = light_field[2]
        push!(wavefront, sum(light_field))
    end
    plt = plot(wavefront, title="Day 1 Light Propagation (integrated density)", xlabel="time step", ylabel="integrated intensity")
    savefig(plt, outpath)
    return (ok=true, path=outpath)
end

# --- Execution block ---
function run_day1(; save_state::Bool = true)
    S = compute_S_day1()
    res = anchor_to_cmb(S)

    state = Dict(
        "day" => 1,
        "S" => string(S),
        "emergence" => res.emergence,
        "light_density_raw" => string(res.light_density_raw),
        "anchored_density_J_per_m3" => string(res.anchored_density),
        "post_D_eff" => string(BigFloat(25) - BigFloat(10))
    )

    # Attempt FIRAS fetch and plot (best-effort)
    firas = try_fetch_firas()
    state["firas_fetch"] = firas
    plot_res = plot_dummy_wavefront()
    state["wavefront_plot"] = plot_res

    if save_state
        if has_json
            open("day1_state.json", "w") do io
                write(io, JSON.json(state; indent=4))
            end
        else
            # Minimal JSON fallback for environments without JSON.jl
            open("day1_state.json", "w") do io
                write(io, "{\n")
                first = true
                for (k,v) in state
                    if !first
                        write(io, ",\n")
                    end
                    write(io, "  \"$(k)\": \"$(v)\"")
                    first = false
                end
                write(io, "\n}\n")
            end
        end
    end

    @printf("Day 1 S: %.10g\n", S)
    @printf("Emergence: %s\n", res.emergence)
    @printf("Anchored radiation density (J/m^3): %s\n", string(res.anchored_density))
    @printf("State saved: day1_state.json\n")
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day1()
end
