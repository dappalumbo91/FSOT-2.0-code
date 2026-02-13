Day 1 — FSOT 2.0: Light Emergence

What this folder contains
- `day1.jl` — Julia module that computes the FSOT Day‑1 scalar (S), anchors the result to observed CMB constants, attempts a best‑effort FIRAS download, and writes `day1_state.json`.

How to run (recommended: Julia 1.8+ / 1.10+)
1. Install Julia (if not already): https://julialang.org/downloads/
2. From the repo root run:
   julia --project=.\DataAnalysisExpert -e "include(\"DataAnalysisExpert/day1/day1.jl\")"

Optional dependencies
- Downloads.jl — used to fetch FIRAS spectrum (best‑effort). Install via Julia REPL: `using Pkg; Pkg.add("Downloads")`
- Plots.jl — used to generate a small wavefront PNG. Install via `using Pkg; Pkg.add("Plots")`

Outputs
- `day1_state.json` — computed S, anchored density, and fetch/plot status.
- `day1_wavefront.png` — optional plot produced if Plots.jl is available.

Download anchors (fetch-on-demand)
- A lightweight download helper is provided: `DataAnalysisExpert/download_anchors.jl` — use this to fetch FIRAS monopole and small Planck plot data on demand (keeps the repo small).
- Recommended anchors (not included in repo):
  - FIRAS monopole spectrum (COBE/FIRAS): https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spectrum/firas_monopole_spec_v1.txt
  - Planck TT/EE/TE small plot data & cosmological params: https://lambda.gsfc.nasa.gov/ and https://pla.esac.esa.int/

Next steps
- Confirm you can run `day1.jl` locally. I can then:
  - Chain Day 1 → Day 2 simulation module in Julia (inflation/firmament).
  - Port Day 1 to Rust / C# / C++ on request.
  - Add automated data pulls from Planck/LAMBDA and formal statistical anchoring via `download_anchors.jl`.

Day 8 — Eden/Fall narrative highlights 🔥
- Files produced by the Day‑8 narrative MC (calibrated):
  - `day8_narrative_results.csv` — per‑epoch timeline rows (reset, strife length, fidelity before/after)
  - `day8_narrative_fidelity_history.csv` — fidelity time series
  - `day8/data/day8_narrative_annotated.csv` — enriched per‑epoch annotations (event_type, dominion proxy, pct change)
  - `day8/data/day8_narrative_highlights.csv` / `day8/json/day8_narrative_highlights.json` — top covenant renewal bursts (ready for X posts)
  - `day8/images/day8_narrative_highlights.png` — fidelity timeline with highlighted covenant bursts
- Quick usage:
  - `julia --project=. tools/run_day8_narrative.jl` — run multi‑epoch narrative (configurable via ENV)
  - `julia --project=. tools/run_export_day8_timeline.jl` — export annotated timeline + highlights
- Recommended: use the `day8/json/day8_narrative_highlights.json` captions directly for X posts (includes short prediction‑style captions).

Sources & anchors
- Planck 2018 cosmological parameters (arXiv:1807.06209)
- FIRAS monopole blackbody results (COBE/FIRAS)

If you want a different language (Rust / C# / C++), tell me which and I'll port the Day‑1 module next.