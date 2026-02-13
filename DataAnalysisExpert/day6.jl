# Day 6 — Land Animals & Humanity in Image (Observed=True)
# Chains Day-5 MC averages into Day-6 land-animal swarming and humanity precursor

using Printf
include(joinpath(@__DIR__, "day1", "day1.jl"))
include("day5.jl")   # provides load_day4_tuned fallback if needed

function load_day5_mc()
    # read from day5_state.json or fallback tuned values
    if isfile("day5_state.json")
        try
            txt = read("day5_state.json", String)
            # very small, safe parse (assumes keys present as strings)
            if occursin("diversity_proxy", txt) && occursin("coherence", txt)
                # reuse tuned defaults if parse fails; this is a scaffold
                return (diversity = BigFloat("2.3e10"), repro_coherence = BigFloat("0.918"), post_D_eff = BigFloat(3), timescale_ga = BigFloat(4.6))
            end
        catch
            # fall through to default
        end
    end
    return (diversity = BigFloat("2.3e10"), repro_coherence = BigFloat("0.918"), post_D_eff = BigFloat(3), timescale_ga = BigFloat(4.6))
end

function compute_S_day6(prev_diversity::BigFloat, prev_D_eff::BigFloat)
    Δψ = BigFloat("3.618") + BigFloat(randn()*0.7)  # Ultimate yang + strife
    observed = true

    # ensure phase_eff constants are defined locally
    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)
    consciousness_factor = observed ? ψ_con * BigFloat("10") : BigFloat(0)

    # growth term scaled to diversity (bounded)
    growth_term = exp(min(prev_diversity / BigFloat(1e9), BigFloat(50))) * log(prev_D_eff + 1)
    poof_term = POOF_FACTOR * exp(min(Δψ, BigFloat(20))) * cos((ψ_con + Δψ)/η_eff)
    suction_term = -SUCTION_FACTOR * BigFloat(0.01)
    quirk_term = consciousness_factor * exp(min(Δψ, BigFloat(10))) * BigFloat(100)

    S = poof_term + suction_term + growth_term + quirk_term
    # clamp for safety
    S = clamp(S, BigFloat(-100.0), BigFloat(1e6))
    return S
end

function humanity_proxy(S::BigFloat, repro_coherence::BigFloat)
    land_animals = S > BigFloat(8)
    humanity = land_animals && (Float64(repro_coherence) > 0.9)
    dominion = humanity ? exp(min(Float64(S), 50.0)) * Float64(repro_coherence) : 1.0
    observed_fidelity = humanity ? "Consciousness_factor full—mid-scale observer active" : "Pre-human"
    return (animals=land_animals, humanity=humanity, dominion=dominion, fidelity=observed_fidelity)
end

# Instincts / Curiosity / Tool Dominion helpers
function curiosity_proxy(strife_factor::BigFloat, consciousness::Real)
    phase_variance = BigFloat(0.15)
    c = BigFloat(consciousness)
    curiosity = Float64(clamp(phase_variance * strife_factor * (c + BigFloat(0.1)), 0.0, 1.0))
    return curiosity
end

function tool_dominion(strife_factor::BigFloat, repro_coherence::Real)
    rc = BigFloat(repro_coherence)
    manipulation = Float64(clamp(exp(min(strife_factor/3, 20.0)) * (rc + BigFloat(0.1)), 1.0, 1e8))
    community = manipulation > 1e6 ? "Cooperative scaling active" : "Individual instinct"
    return (manipulation=manipulation, community=community)
end

function survival_fittest(strife_factor::BigFloat, community::Real)
    comm = Float64(community)
    fittest = (strife_factor > 3.0 && strife_factor < 8.0) || (comm > 1e6)
    volatility = strife_factor > 8.0 ? "High—damped by community power" : "Balanced"
    stagnation = strife_factor < 2.0 ? "Low—weakened by curiosity variance" : "Balanced"
    return (fittest=fittest, volatility=volatility, stagnation=stagnation)
end

function run_day6()
    d5 = load_day5_mc()
    S = compute_S_day6(d5.diversity, d5.post_D_eff)
    res = humanity_proxy(S, d5.repro_coherence)

    state = Dict(
        "day" => 6,
        "S" => string(S),
        "land_animals" => string(res.animals),
        "humanity_image" => string(res.humanity),
        "dominion_proxy" => string(res.dominion),
        "fidelity" => res.fidelity
    )

    # write JSON fallback
    open("day6_state.json", "w") do io
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

    @printf("Day 6 S: %.10g\n", S)
    @printf("Land Animals: %s\n", res.animals)
    @printf("Humanity in Image: %s\n", res.humanity)
    @printf("Dominion proxy: %.6g\n", res.dominion)
    @printf("Fidelity: %s\n", res.fidelity)

    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day6()
end
