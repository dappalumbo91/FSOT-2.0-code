# Runner for exporting annotated Day-8 timeline for highlights analysis
include(joinpath(@__DIR__, "..", "day8_narrative.jl"))

# default input files produced by narrative run
narr = "day8/data/day8_narrative_results.csv"
fid = "day8/data/day8_narrative_fidelity_history.csv"
out_annot = "day8/data/day8_narrative_annotated.csv"
out_hl = "day8/data/day8_narrative_highlights.csv"
out_png = "day8/images/day8_narrative_highlights.png"

# run Python exporter
try
    run(`python tools/export_day8_timeline.py $(narr) $(fid) $(out_annot) $(out_hl) $(out_png)`)
catch err
    @warn "Failed to export Day-8 timeline" err
end
println("Export complete: ", out_annot, ", ", out_hl, ", ", out_png)
