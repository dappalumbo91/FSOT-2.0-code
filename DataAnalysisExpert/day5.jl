# Day 5 — Swarming Creatures & Observed Humanity Precursor (scaffold)
using Printf

function load_day4_tuned()
    # read aggregated Day‑4 MC summary if present, else default tuned values
    path = "day4_mc_summary.csv"
    if isfile(path)
        lines = readlines(path)
        if length(lines) >= 2
            cols = split(lines[2], ',')
            return (regularity = BigFloat(parse(Float64, cols[4])),
                    dna_complexity = BigFloat(parse(Float64, cols[5])),
                    post_D_eff = BigFloat(4),
                    time_years = BigFloat(parse(Float64, cols[2])))
        end
    end
    return (regularity = BigFloat("99.999999"), dna_complexity = BigFloat("1e6"), post_D_eff = BigFloat(4), time_years = BigFloat("1e10"))
end

function compute_S_day5(prev_complexity::BigFloat, prev_D_eff::BigFloat)
    Δψ = BigFloat("2.618") + BigFloat(randn()*0.5)  # Strong yang swarming
    observed = true
    consciousness_factor = observed ? ψ_con : BigFloat(0)

    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)

    growth_term = exp(prev_complexity / BigFloat(1e5)) * log(prev_D_eff + 1)
    poof_term = POOF_FACTOR * exp(min(Δψ, BigFloat(20))) * cos((ψ_con + Δψ)/η_eff)
    suction_term = -SUCTION_FACTOR * BigFloat(0.05)
    quirk_term = observed ? consciousness_factor * exp(min(Δψ, BigFloat(10))) * BigFloat(10) : BigFloat("0.618")

    S = poof_term + suction_term + growth_term + quirk_term
    return S
end

function biota_proxy(S::BigFloat, regularity::BigFloat)
    swarming = S > BigFloat(5)
    diversity = swarming ? exp(min(Float64(S), 20.0)) * Float64(regularity) / 100.0 : 1.0
    observed_coherence = swarming ? "Humanity precursor active" : "Pre-observed"
    return (swarming=swarming, diversity=diversity, coherence=observed_coherence)
end

function run_day5()
    d4 = load_day4_tuned()
    S = compute_S_day5(d4.dna_complexity, d4.post_D_eff)
    res = biota_proxy(S, d4.regularity)

    state = Dict("day"=>5, "S"=>string(S), "swarming"=>string(res.swarming),
                 "diversity_proxy"=>string(res.diversity), "coherence"=>res.coherence)

    open("day5_state.json", "w") do io
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

    @printf("Day 5 S: %.10g\n", S)
    @printf("Swarming Biota: %s\n", res.swarming)
    @printf("Diversity Proxy: %.6g\n", res.diversity)
    @printf("Coherence: %s\n", res.coherence)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day5()
end
