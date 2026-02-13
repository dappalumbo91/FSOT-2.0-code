# Runner: Day 1 -> Day 2 -> Cone overlay with logging
include(joinpath(@__DIR__, "..", "day1", "day1.jl"))
include("day2.jl")
include(joinpath(@__DIR__, "..", "day1", "cone_overlay.jl"))

function run_all_and_log(logpath::String = "day1_2_cones_log.txt")
    open(logpath, "w") do io
        println(io, "== FSOT Day1 -> Day2 -> Cone Overlay Log ==")
        println(io, "Running Day 1...")
        d1 = run_day1()
        println(io, "Day 1 output:")
        for (k,v) in d1
            println(io, "  $k => $v")
        end

        println(io, "\nRunning Day 2...")
        d2 = run_day2()
        println(io, "Day 2 output:")
        for (k,v) in d2
            println(io, "  $k => $v")
        end

        println(io, "\nRunning Cone overlay (Day 1 anchored)...")
        cone_res = run_cone_overlay(save_log=false)
        println(io, "Cone overlay output summary:")
        @printf(io, "  T_eff_K = %.2f\n", cone_res["T_eff_K"])
        nr = cone_res["norm_responses"]
        @printf(io, "  norm responses (L,M,S) = %.4f, %.4f, %.4f\n", nr[:L], nr[:M], nr[:S])

        println(io, "\nAll steps complete. Files produced: day1_state.json, day2_state.json, cone_overlay_log.txt")
    end
    println(read(logpath, String))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_and_log()
end
