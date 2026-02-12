ENV["SWEEP_TRIALS"] = "4"
include(joinpath(@__DIR__, "..", "test", "test_sensitivity_sweep.jl"))
println("sensitivity-sweep test run complete")
