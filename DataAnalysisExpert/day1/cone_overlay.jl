# Cone‑sensitivity overlay simulation (FSOT → cone spectra)
# - Computes an effective blackbody spectrum from Day‑1 anchored density
# - Applies Gaussian approximations for L/M/S cone sensitivities
# - Returns normalized cone responses and writes a short report

using Printf
using LinearAlgebra

# Physical constants (SI)
const h = 6.62607015e-34     # Planck constant (J s)
const c = 2.99792458e8       # speed of light (m/s)
const kB = 1.380649e-23      # Boltzmann constant (J/K)
const a_rad = 7.5657e-16     # Radiation constant (J m^-3 K^-4)

# Wavelength grid (meters)
const λ_nm = collect(380.0:2.0:780.0)            # nm
const λ = Float64.(λ_nm) .* 1e-9                 # m

# Planck spectral radiance (per wavelength) — returns spectral energy density proxy
function planck_spectral_radiance(λ::Vector{Float64}, T::Float64)
    # Use spectral radiance B_λ(λ,T) ~ (2hc^2 / λ^5) * 1/(exp(hc/(λ kT)) - 1)
    B = similar(λ)
    for i in eachindex(λ)
        x = h * c / (λ[i] * kB * T)
        # avoid overflow for very small λ*T
        ex = exp(x)
        B[i] = (2 * h * c^2) / (λ[i]^5 * (ex - 1.0))
    end
    return B
end

# Gaussian cone sensitivity approximations (peaks in nm, sigma in nm)
function cone_sensitivity(λ_nm::Vector{Float64}; peak_nm::Float64=560.0, sigma_nm::Float64=40.0)
    s = exp.(-0.5 .* ((λ_nm .- peak_nm) ./ sigma_nm).^2)
    return s ./ maximum(s)
end

# Convert radiation energy density (u, J/m^3) -> effective blackbody temperature
function density_to_T(u::Float64)
    # u = a * T^4
    if u <= 0
        return 2.72548  # fallback to CMB if zero/negative
    end
    return (u / a_rad)^(1/4)
end

# Compute cone responses given anchored_density_J_per_m3 (string or number)
function compute_cone_responses(anchored_density, light_density_raw=nothing)
    u = parse(Float64, string(anchored_density))
    T = density_to_T(u)

    use_perceptual_proxy = false
    B = zeros(length(λ))

    # If physical T is extremely low (no visible content), provide both physical result and
    # a perceptual visible proxy (for visualization/interpretation).
    if T < 100.0
        use_perceptual_proxy = true
        # physical spectrum (very cold) for completeness
        B_phys = planck_spectral_radiance(λ, T)
        # perceptual proxy: use a Sun-like blackbody to show cone overlay behavior
        T_vis = 5500.0
        B_vis = planck_spectral_radiance(λ, T_vis)
        # normalize B_vis for response computation (independent of absolute energy)
        if trapz(λ, B_vis) > 0
            B = B_vis ./ trapz(λ, B_vis)
        else
            B .= 0.0
        end
    else
        B = planck_spectral_radiance(λ, T)
        if trapz(λ, B) > 0
            B .= B ./ trapz(λ, B)
        end
    end

    # Cone curves
    L_curve = cone_sensitivity(λ_nm; peak_nm=560.0, sigma_nm=40.0)
    M_curve = cone_sensitivity(λ_nm; peak_nm=530.0, sigma_nm=40.0)
    S_curve = cone_sensitivity(λ_nm; peak_nm=420.0, sigma_nm=30.0)

    # Responses = integral of spectrum * cone sensitivity
    rL = trapz(λ, B .* L_curve)
    rM = trapz(λ, B .* M_curve)
    rS = trapz(λ, B .* S_curve)

    # Safe normalization
    maxr = maximum([rL, rM, rS, eps()])
    rLn, rMn, rSn = rL/maxr, rM/maxr, rS/maxr

    return Dict(
        "T_eff_K" => T,
        "using_perceptual_proxy" => use_perceptual_proxy,
        "raw_responses" => (L=rL, M=rM, S=rS),
        "norm_responses" => (L=rLn, M=rMn, S=rSn),
        "spectrum_lambda_nm" => λ_nm,
        "spectrum_B_norm" => B
    )
end

# Simple trapezoidal integrator
function trapz(x::Vector{Float64}, y::Vector{Float64})
    s = 0.0
    for i in 1:(length(x)-1)
        s += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    end
    return s
end

# Load day1_state.json (supports JSON.jl or simple fallback)
function load_day1_state(path::String = "day1_state.json")
    if !isfile(path)
        error("day1_state.json not found — run day1/day1.jl first")
    end
    content = read(path, String)
    # Try JSON.jl
    has_json = try
        @eval using JSON; true
    catch
        false
    end
    if has_json
        st = JSON.parse(content)
        return st
    else
        # parse simple fallback format
        m = Dict{String,String}()
        for line in split(content, '\n')
            if occursin(":", line)
                parts = split(line, ":")
                k = replace(strip(parts[1]), '"' => "")
                v = strip(join(parts[2:end], ":"))
                v = replace(v, '"' => "")
                v = replace(v, ',' => "")
                m[k] = v
            end
        end
        return m
    end
end

# Run overlay and print/log results
function run_cone_overlay(; day1_state_path::String = "day1_state.json", save_log::Bool = true)
    st = load_day1_state(day1_state_path)
    anchored_key = haskey(st, "anchored_density_J_per_m3") ? "anchored_density_J_per_m3" : "anchored_density"
    anchored_density = st[anchored_key]

    res = compute_cone_responses(anchored_density)

    msg = IOBuffer()
    @printf(msg, "Cone overlay — derived effective temperature: %.2f K\n", res["T_eff_K"])
    nr = res["norm_responses"]
    @printf(msg, "Normalized cone responses → L: %.4f, M: %.4f, S: %.4f\n", nr[:L], nr[:M], nr[:S])
    @printf(msg, "Anchored density used: %s J/m^3\n", string(anchored_density))

    if save_log
        open("cone_overlay_log.txt", "w") do io
            write(io, String(take!(msg)))
        end
    end

    print(String(take!(msg)))
    return res
end

# If run directly, execute with local day1_state.json
if abspath(PROGRAM_FILE) == @__FILE__
    run_cone_overlay()
end
