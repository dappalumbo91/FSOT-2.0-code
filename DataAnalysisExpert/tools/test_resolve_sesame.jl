include(joinpath(@__DIR__, "..", "planetary_gaia.jl"))
for name in ("Kepler-186", "Kepler-62", "TRAPPIST-1")
    println("Resolving: ", name)
    ra,dec = resolve_coords_via_sesame(name)
    println(" -> ", ra, ", ", dec)
end
