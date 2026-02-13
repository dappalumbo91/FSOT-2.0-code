using CSV, DataFrames, Statistics

fn = joinpath(@__DIR__, "..", "sensitivity", "day4_diagnostic_m0p75_50000.csv")
df = CSV.File(fn) |> DataFrame
n = nrow(df)
pass = df[df.habitable .== true, :]
fail = df[df.habitable .== false, :]
println("Trials = ", n)
println("Habitable fraction = ", round(100* nrow(pass)/n, digits=4), "%")
println("Fail count = ", nrow(fail))
println("Fail convection mean = ", mean(fail.convection), " | Pass convection mean = ", mean(pass.convection))
println("Fail regularity mean = ", mean(fail.regularity), " | Pass regularity mean = ", mean(pass.regularity))
println("Fail S mean = ", mean(fail.S), " | Pass S mean = ", mean(pass.S))
println("Fail dna_folding rate = ", round(100*mean(fail.dna_folding), digits=4), "% | Pass dna_folding rate = ", round(100*mean(pass.dna_folding), digits=4), "%")
println("Fail mean ecc = ", mean(fail.ecc), " | Pass mean ecc = ", mean(pass.ecc))
