# Fetch Paleobiology DB hominin occurrences, fit simple priors, set Day-6 priors, run Day-6 MC (10k)
include(joinpath(@__DIR__, "..", "planetary_db.jl"))
include(joinpath(@__DIR__, "..", "day6", "day6_mc.jl"))

println("Calibrating hominid priors from Paleobiology DB (genus=Homo)")
jsonp = joinpath(@__DIR__, "..", "data", "paleobio_homo.json")
mkpath(joinpath(@__DIR__, "..", "data"))
fetch_paleobio_hominins(jsonp, genus="Homo", limit=1000)
summary = summarize_paleobio_ages(jsonp)
println("Paleobio Homo summary: ", summary)

# simple mapping from occurrence ages to priors
# if data insufficient, use defaults already in HOMINID_PRIORS
if summary[:n] >= 10 && !isnan(summary[:mean])
    # use mean occurrence age as proxy for earliest genus presence (coarse)
    fitted_homo = max(0.2, summary[:mean])
    # divergence prior: add conservative offset (~3-4 Ma older)
    fitted_div = fitted_homo + 4.0
    # sapiens prior: use literature default (PCR not well sampled in Paleobio)
    fitted_sapiens = 0.3
    fitted_behavior = 0.05
    pri = Dict(:divergence=>fitted_div, :homo=>fitted_homo, :sapiens=>fitted_sapiens, :behavior=>fitted_behavior)
    println("Fitted hominid priors (Ma): ", pri)
    set_hominid_priors!(pri)
    # persist priors
    open(joinpath(@__DIR__, "..", "data", "hominid_priors.json"), "w") do io
        println(io, JSON.json(pri))
    end
else
    println("Insufficient Paleobio samples — leaving default priors unchanged.")
end

# run enriched Day-6 MC (10k)
println("Running Day-6 MC (10k) with calibrated priors...")
include(joinpath(@__DIR__, "..", "tools", "run_day6_enriched.jl"))
