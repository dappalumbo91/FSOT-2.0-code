# orbital proxy helpers for Day-4 MC
function orbital_proxy(regularity::Real, convection::Real)
    r = Float64(regularity)
    c = Float64(convection)
    eccentricity = clamp(1.0 - r/100.0, 0.0, 1.0)
    coherence = c > 0.8 ? "Keplerian stable" : "Chaotic migration"
    habitable = eccentricity < 0.05 && c > 0.5
    return (eccentricity=eccentricity, coherence=coherence, habitable=habitable)
end
