# Runner for publication-grade sensitivity sweep
# - sets trials=50000 and enables extended axes (metallicity->dynamo)
ENV["SWEEP_TRIALS"] = "50000"
ENV["SWEEP_EXTENDED"] = "true"
println("Starting full sensitivity sweep: trials=50000, extended axes ON")
include(joinpath(@__DIR__, "sensitivity_sweep.jl"))
println("Full sweep finished")
