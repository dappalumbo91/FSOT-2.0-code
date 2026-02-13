# Day-6 MC — Land Animals, Humanity Image & Consciousness stability MC
# Chains Day-5 MC averages into Day-6 outcomes; tracks humanity-image rates + dominion proxy

using Random, Statistics, Printf
include(joinpath(@__DIR__, "..", "day6.jl"))

function compute_S_day6_mc(diversity::BigFloat, repro_coherence::Float64)
    Δψ = BigFloat("3.618") + BigFloat(randn()*0.6)
    phase_variance = BigFloat("0.15")
    η_eff = φ * (1 + phase_variance)

    growth_term = exp(min(diversity / BigFloat(1e9), BigFloat(50))) * BigFloat(0.1)
    poof_term = POOF_FACTOR * exp(min(Δψ, BigFloat(20))) * cos((ψ_con + Δψ)/η_eff)
    suction_term = -SUCTION_FACTOR * BigFloat(0.01)
    quirk_term = (ψ_con * BigFloat(10)) * exp(min(Δψ, BigFloat(10))) * BigFloat(50)

    S = poof_term + suction_term + growth_term + quirk_term
    return clamp(S, BigFloat(-100.0), BigFloat(1e6))
end

# --- hominid timeline calibration helpers (fossil priors + modulation) ---
# module-level adjustable priors (Ma)
const HOMINID_PRIORS = Dict(:divergence => 6.5, :homo => 2.0, :sapiens => 0.3, :behavior => 0.05)

function set_hominid_priors!(p::Dict)
    for k in keys(HOMINID_PRIORS)
        if haskey(p, k)
            HOMINID_PRIORS[k] = p[k]
        end
    end
    return HOMINID_PRIORS
end

function _dominon_score(dom::Float64)
    if dom <= 0.0 || isnan(dom)
        return 0.4
    end
    # map large dominion proxy (~1e21) into a 0.0-2.0 scale
    ds = clamp(log10(dom + 1.0)/21.0, 0.0, 2.0)
    return ds
end

function calibrate_hominid_timeline(strife::Float64, dominion::Float64, curiosity::Float64, community::Bool; rng::AbstractRNG=Random.GLOBAL_RNG)
    # fossil priors (Ma)
    base_divergence = 6.5   # great apes / hominins divergence ~6-7 Ma
    base_homo = 2.0         # Homo genus ~2 Ma
    base_sapiens = 0.3      # anatomically modern sapiens ~0.3 Ma
    base_behavior = 0.05    # behavioral modernity ~50 ka

    # capability score from dominion/curiosity/community
    dscore = _dominon_score(dominion)
    capability = 0.4*dscore + 0.4*clamp(curiosity, 0.0, 2.0) + 0.2*(community ? 1.0 : 0.0)

    # strife optimal around ~5 (moderate stress -> faster innovation), penalize extremes
    strife_opt = 5.0
    strife_penalty = exp(-((strife - strife_opt)^2)/18.0)  # in (0,1]
    effect = capability * strife_penalty

    # acceleration factor: >1 speeds evolution, <1 slows it
    accel = clamp(1.0 + 0.6*(effect - 0.5), 0.5, 2.0)

    # draw noisy event times (Gaussian around scaled priors)
    # draw approximate Gaussian samples using randn
    divergence = max(0.1, (base_divergence / accel) + randn(rng)*(0.8/accel))
    homo_genus = max(0.01, (base_homo / accel) + randn(rng)*(0.4/accel))
    sapiens = max(0.001, (base_sapiens / accel) + randn(rng)*(0.12/accel))
    behavior = max(0.0001, (base_behavior / accel) + randn(rng)*(0.02/accel))

    # probabilistic milestone attainment (higher effect -> higher chance)
    p_div = clamp(0.25 + 0.75*effect, 0.0, 1.0)
    p_hg = clamp(0.2 + 0.75*effect, 0.0, 1.0)
    p_sp = clamp(0.15 + 0.8*effect, 0.0, 1.0)
    p_bm = clamp(0.12 + 0.8*effect, 0.0, 1.0)

    # sample attainment
    div_flag = rand(rng) < p_div
    hg_flag = rand(rng) < p_hg
    sp_flag = rand(rng) < p_sp
    bm_flag = rand(rng) < p_bm

    # if event not attained, set NaN timing
    divergence = div_flag ? divergence : NaN
    homo_genus = hg_flag ? homo_genus : NaN
    sapiens = sp_flag ? sapiens : NaN
    behavior = bm_flag ? behavior : NaN

    return Dict(:divergence_Ma=>divergence, :homo_genus_Ma=>homo_genus, :sapiens_Ma=>sapiens, :behavioral_Ma=>behavior,
                :divergence_flag=>div_flag, :homo_flag=>hg_flag, :sapiens_flag=>sp_flag, :behavior_flag=>bm_flag,
                :accel=>accel, :capability=>capability, :effect=>effect)
end

function run_day6_mc(; trials::Int=10000, rng_seed::Int=42, day5_path::AbstractString="day5_mc_results.csv")
    Random.seed!(rng_seed)
    # load Day-5 aggregates — prefer an explicit MC CSV if provided, else fallback to load_day5_mc()
    if isfile(day5_path)
        try
            tbl = CSV.File(day5_path) |> DataFrame
            # diversity: mean of 'diversity' column
            diversity = mean(Float64.(filter(!ismissing, tbl[:diversity])))
            # repro_coherence: fraction of rows with 'Sperm/egg fidelity active'
            if :coherence in names(tbl)
                repro_frac = count(x->strip(String(x))=="Sperm/egg fidelity active", tbl[:, :coherence]) / nrow(tbl)
                repro = clamp(repro_frac, 0.0, 1.0)
            else
                repro = 0.918
            end
        catch
            # fallback to original loader
            d5 = load_day5_mc()
            diversity = Float64(d5.diversity)
            repro = Float64(d5.repro_coherence)
        end
    else
        d5 = load_day5_mc()
        diversity = Float64(d5.diversity)
        repro = Float64(d5.repro_coherence)
    end

    rows = Vector{Dict{String,Any}}()
    human_count = 0
    land_count = 0
    dom_vals = Float64[]
    Svals = Float64[]

    for i in 1:trials
        # small jitter in diversity to model ecological variance
        div_j = diversity * (0.8 + 0.4*rand())
        S = compute_S_day6_mc(BigFloat(div_j), repro)

        # extended proxies: curiosity, tool dominion, survival/fittest
        # derive a strife_factor proxy from S and diversity
        strife_factor = clamp(Float64(abs(S) / 10.0), 0.0, 20.0)
        curiosity = curiosity_proxy(BigFloat(strife_factor), Float64(repro))
        tool = tool_dominion(BigFloat(strife_factor), Float64(repro))
        surv = survival_fittest(BigFloat(strife_factor), Float64(tool.manipulation))

        res = humanity_proxy(S, BigFloat(repro))

        # calibrate hominid timelines for this trial
        community_bool = isa(tool.community, String) ? occursin("Cooperative", tool.community) : Bool(tool.community)
        hom = calibrate_hominid_timeline(strife_factor, Float64(res.dominion), Float64(curiosity), community_bool)

        push!(rows, Dict("trial"=>i,
                        "S"=>Float64(S),
                        "land"=>res.animals,
                        "humanity"=>res.humanity,
                        "dominion"=>Float64(res.dominion),
                        "fidelity"=>res.fidelity,
                        "curiosity"=>Float64(curiosity),
                        "manipulation"=>Float64(tool.manipulation),
                        "community_flag"=>tool.community,
                        "fittest"=>surv.fittest,
                        "divergence_Ma"=>hom[:divergence_Ma],
                        "homo_genus_Ma"=>hom[:homo_genus_Ma],
                        "sapiens_Ma"=>hom[:sapiens_Ma],
                        "behavioral_Ma"=>hom[:behavioral_Ma],
                        "divergence_flag"=>hom[:divergence_flag],
                        "homo_flag"=>hom[:homo_flag],
                        "sapiens_flag"=>hom[:sapiens_flag],
                        "behavior_flag"=>hom[:behavior_flag],
                        "hominid_accel"=>hom[:accel],
                        "hominid_capability"=>hom[:capability]))

        push!(Svals, Float64(S))
        if res.humanity
            human_count += 1
            push!(dom_vals, Float64(res.dominion))
        end
        if res.animals
            land_count += 1
        end
    end

    human_rate = human_count / trials * 100
    land_rate = land_count / trials * 100
    mean_dom = isempty(dom_vals) ? 0.0 : mean(dom_vals)

    # write CSV
    csvp = "day6_mc_results.csv"
    open(csvp, "w") do io
        println(io, "trial,S,land,humanity,dominion,fidelity,curiosity,manipulation,community_flag,fittest,divergence_Ma,homo_genus_Ma,sapiens_Ma,behavioral_Ma,divergence_flag,homo_flag,sapiens_flag,behavior_flag,hominid_accel,hominid_capability")
        for r in rows
            println(io, string(r["trial"], ",", r["S"], ",", r["land"], ",", r["humanity"], ",", r["dominion"], ",\"", r["fidelity"], "\",", r["curiosity"], ",", r["manipulation"], ",\"", r["community_flag"], "\",", r["fittest"], ",", r["divergence_Ma"], ",", r["homo_genus_Ma"], ",", r["sapiens_Ma"], ",", r["behavioral_Ma"], ",", r["divergence_flag"], ",", r["homo_flag"], ",", r["sapiens_flag"], ",", r["behavior_flag"], ",", r["hominid_accel"], ",", r["hominid_capability"]))
        end
    end

    # also write a dedicated hominid CSV for downstream analysis
    hom_csv = "day6_mc_hominid.csv"
    open(hom_csv, "w") do io
        println(io, "trial,divergence_Ma,homo_genus_Ma,sapiens_Ma,behavioral_Ma,divergence_flag,homo_flag,sapiens_flag,behavior_flag,hominid_accel,hominid_capability")
        for r in rows
            println(io, string(r["trial"], ",", r["divergence_Ma"], ",", r["homo_genus_Ma"], ",", r["sapiens_Ma"], ",", r["behavioral_Ma"], ",", r["divergence_flag"], ",", r["homo_flag"], ",", r["sapiens_flag"], ",", r["behavior_flag"], ",", r["hominid_accel"], ",", r["hominid_capability"]))
        end
    end

    # write log
    open("day6_mc_log.txt", "w") do io
        @printf(io, "Day-6 MC — trials=%d, seed=%d\n", trials, rng_seed)
        @printf(io, "Land animals rate: %.3f%%\n", land_rate)
        @printf(io, "Humanity-image rate: %.3f%%\n", human_rate)
        @printf(io, "Mean dominion proxy (human worlds): %.6g\n", mean_dom)
        @printf(io, "Mean manipulation (manipulation): %.6g\n", mean([r["manipulation"] for r in rows]))
        @printf(io, "Curiosity mean: %.6g\n", mean([r["curiosity"] for r in rows]))
        @printf(io, "Survival fittest fraction: %.6g\n", count(r->r["fittest"]==true, rows)/length(rows))
    end

    # summary stdout
    @printf("Day-6 MC summary — trials=%d\nLand animals: %.3f%% | Humanity-image: %.3f%% | Mean dominion: %.6g\n",
            trials, land_rate, human_rate, mean_dom)

    return Dict("trials"=>trials, "land_rate"=>land_rate, "humanity_rate"=>human_rate, "mean_dominion"=>mean_dom)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_day6_mc(trials=10000, rng_seed=42)
end
