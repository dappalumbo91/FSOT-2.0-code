# day2_sweep.jl — parameter sweep + calibration for Day‑2 BH‑valve proxies
# Outputs a CSV with raw and calibrated proxy statistics for each parameter combo.

using Printf, Random, Statistics
include("day2_mc_extended.jl")   # exposes compute_S_day2_bh_mode and anchor helpers

# Calibration defaults (tunable)
const OUTGAS_MASS_SCALE = 1e-4    # convert raw outgas proxy -> M_sun
const FRB_ENERGY_SCALE = 1e33     # convert raw FRB proxy -> erg
const LENSING_SCALE = 1e-7        # convert raw lensing proxy -> arcsec

# Calibrate raw scale_growth -> e-folds (log10 mapping, clamped)
function calibrate_e_folds(raw_scale_growth::Float64)
    ef = log10(1.0 + raw_scale_growth)
    return clamp(ef, 0.0, 300.0)
end

# Run MC for a single parameter set (uses compute_S_day2_bh_mode)
function run_mc_for_params(delta_psi_base_f::Float64, valve_scale::Float64, puncture_thresh::Float64; trials::Int=2000, seed::Int=42)
    Random.seed!(seed)
    day1 = load_day1_state()
    prev_D_eff = haskey(day1, "post_D_eff") ? parse(Float64, day1["post_D_eff"]) : 15.0

    separations = 0
    punctures = 0
    outgasses = Float64[]
    frb_energies = Float64[]
    lensings = Float64[]
    scale_raws = Float64[]

    for i in 1:trials
        res = compute_S_day2_bh_mode(prev_D_eff, valve_scale; delta_psi_base_f=delta_psi_base_f)
        # apply puncture threshold override
        punct = (abs(res.S) > puncture_thresh) && (res.S < 0.0)
        if res.separation
            separations += 1
        end
        if punct
            punctures += 1
            push!(outgasses, res.outgas)
            push!(frb_energies, res.frb_proxy)
        end
        if res.lensing_proxy > 0
            push!(lensings, res.lensing_proxy)
        end
        push!(scale_raws, anchor_to_expansion_refined_f(res.S))
    end

    sep_rate = separations / trials * 100
    punc_rate = punctures / trials * 100
    mean_outgas_raw = length(outgasses) > 0 ? mean(outgasses) : 0.0
    mean_frb_raw = length(frb_energies) > 0 ? mean(frb_energies) : 0.0
    mean_lens_raw = length(lensings) > 0 ? mean(lensings) : 0.0
    mean_scale_raw = mean(scale_raws)

    # Calibrate to physical units
    mean_outgas_Msun = mean_outgas_raw * OUTGAS_MASS_SCALE
    mean_frb_erg = mean_frb_raw * FRB_ENERGY_SCALE
    mean_lensing_arcsec = mean_lens_raw * LENSING_SCALE
    mean_e_folds = calibrate_e_folds(mean_scale_raw)

    return Dict(
        :delta_psi_base => delta_psi_base_f,
        :valve_scale => valve_scale,
        :puncture_thresh => puncture_thresh,
        :sep_rate => sep_rate,
        :punc_rate => punc_rate,
        :mean_outgas_raw => mean_outgas_raw,
        :mean_outgas_Msun => mean_outgas_Msun,
        :mean_frb_raw => mean_frb_raw,
        :mean_frb_erg => mean_frb_erg,
        :mean_lens_raw => mean_lens_raw,
        :mean_lensing_arcsec => mean_lensing_arcsec,
        :mean_scale_raw => mean_scale_raw,
        :mean_e_folds => mean_e_folds,
        :trials => trials
    )
end

# Parameter grid and sweep (default modest grid)
delta_psi_grid = [-1.2, -1.8, -2.618]
valve_scale_grid = [1.0, 10.0, 100.0]
puncture_thresh_grid = [3.0, 4.0, 5.0]
trials_per_combo = 2000

out_csv = "day2/data/day2_sweep_results.csv"
open(out_csv, "w") do io
    header = ["delta_psi_base","valve_scale","puncture_thresh","sep_rate","punc_rate",
              "mean_outgas_raw","mean_outgas_Msun","mean_frb_raw","mean_frb_erg",
              "mean_lens_raw","mean_lensing_arcsec","mean_scale_raw","mean_e_folds","trials"]
    println(io, join(header, ","))
    for dp in delta_psi_grid, vs in valve_scale_grid, pt in puncture_thresh_grid
        res = run_mc_for_params(dp, vs, pt; trials=trials_per_combo, seed=42)
        row = [string(res[k]) for k in (:delta_psi_base,:valve_scale,:puncture_thresh,:sep_rate,:punc_rate,
                                         :mean_outgas_raw,:mean_outgas_Msun,:mean_frb_raw,:mean_frb_erg,
                                         :mean_lens_raw,:mean_lensing_arcsec,:mean_scale_raw,:mean_e_folds,:trials)]
        println(io, join(row, ","))
        @printf("Completed combo dp=%.3f vs=%.1f pt=%.1f -> sep=%.2f%% punc=%.2f%%\n", dp, vs, pt, res[:sep_rate], res[:punc_rate])
    end
end

println("Sweep complete — results written to $out_csv")
