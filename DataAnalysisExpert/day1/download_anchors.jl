# download_anchors.jl — Best-effort fetch for Day‑1 anchoring data
# Usage: julia download_anchors.jl

using Downloads
using Printf

mkpath("data")

function download_firas()
    urls = [
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spectrum/firas_monopole_spec_v1.txt",
        "https://lambda.gsfc.nasa.gov/data/cobe/firas/monopole_spectrum/firas_monopole_spec.txt"
    ]
    for u in urls
        try
            Downloads.download(u, "data/firas_monopole_spec.txt")
            @printf("FIRAS fetched: %s\n", u)
            return true
        catch e
            @printf("FIRAS fetch failed for %s — %s\n", u, e)
        end
    end
    @printf("FIRAS fetch failed — use manual download or check network/URL.\n")
    return false
end

function download_planck_tt_plot()
    # Small/likely locations for Planck TT plot data (may change on source sites)
    urls = [
        "https://lambda.gsfc.nasa.gov/graphics/planck/pr4_tt_spectrum.txt",
        "https://lambda.gsfc.nasa.gov/graphics/planck/pr2_tt_spectrum.txt"
    ]
    for u in urls
        try
            Downloads.download(u, "data/planck_tt_spectrum.txt")
            @printf("Planck TT plot data fetched: %s\n", u)
            return true
        catch e
            @printf("Planck TT fetch failed for %s — %s\n", u, e)
        end
    end
    @printf("Planck TT fetch failed — check LAMBDA graphics pages manually.\n")
    return false
end

println("Downloading anchoring assets into DataAnalysisExpert/data (best-effort)")
download_firas()
download_planck_tt_plot()
println("Done.  If fetches failed, please download the small files manually and place them in DataAnalysisExpert/data/")
