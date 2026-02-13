include("../day1/day1.jl")
for f in ("test_data/planck_real.fits", "test_data/planck_map.fits", "test_data/planck_params.fits")
    println("--- ", f)
    hdr = parse_fits_header(f)
    println("HEADER KEYS: ", collect(keys(hdr)))
    println("HEADER SIMPLE: ", get(hdr, "SIMPLE", "<missing>"))
    println("TELESCOP: ", get(hdr, "TELESCOP", "<missing>"))
    println()
end
