# day2_visualize.jl — visualize day2 sweep results
# Produces: puncture_heatmap.png, frb_energy_hist.png, quasar_scatter.png

try
    if get(ENV, "CI_SKIP_PLOTTING", "0") == "1"
        @info "CI_SKIP_PLOTTING=1 — skipping Plots import in day2_visualize"
    else
        using Plots, CSV, DataFrames, Statistics
    end
catch
    println("Required plotting packages not installed. Install Plots, CSV, DataFrames to generate PNGs.")
    return
end

# prefer GR backend for portability
gr()

csv_path = "day2/data/day2_sweep_results.csv"
if !isfile(csv_path)
    error("Sweep CSV not found: $csv_path — run day2_sweep.jl first")
end

df = CSV.File(csv_path) |> DataFrame

# Ensure columns are present
required = [:delta_psi_base, :valve_scale, :puncture_thresh, :punc_rate, :mean_frb_erg, :mean_outgas_Msun, :mean_lensing_arcsec, :mean_e_folds]
for c in required
    if !haskey(df, Symbol(c))
        error("Missing column in CSV: $c")
    end
end

# Heatmap: puncture probability vs Δψ (rows) and valve_scale (cols) for a fixed puncture_thresh (use first)
unique_dp = unique(df.delta_psi_base)
unique_vs = unique(df.valve_scale)
pt_values = unique(df.puncture_thresh)
selected_pt = pt_values[1]

# build matrix
mat = zeros(Float64, length(unique_dp), length(unique_vs))
for (i, dp) in enumerate(unique_dp)
    for (j, vs) in enumerate(unique_vs)
        row = filter(r -> r.delta_psi_base == dp && r.valve_scale == vs && r.puncture_thresh == selected_pt, df)
        mat[i, j] = isempty(row) ? NaN : Float64(row.punc_rate[1])
    end
end

heatmap(unique_dp, unique_vs, mat', xlabel="Δψ (yin bias)", ylabel="Valve scale", title="Puncture probability (pt=$selected_pt)", color=:viridis, clims=(0,100))
savefig("day2/images/puncture_heatmap.png")

# Histogram: calibrated FRB energies (erg)
frb_vals = Float64.(df.mean_frb_erg)
histogram(frb_vals, bins=20, xlabel="Energy (erg)", ylabel="Count", title="Calibrated FRB energy (mean per combo)", label=false)
savefig("day2/images/frb_energy_hist.png")

# Scatter: Outgas mass vs e-folds
scatter(df.mean_outgas_Msun, df.mean_e_folds, xlabel="Mean outgas mass (M_sun)", ylabel="e-folds proxy (calibrated)", title="Outgas mass vs expansion proxy", label=false)
savefig("day2/images/quasar_scatter.png")

println("Saved: day2/images/puncture_heatmap.png, day2/images/frb_energy_hist.png, day2/images/quasar_scatter.png")
