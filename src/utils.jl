_t₁(x, df::Float64) = pdf(TDist(df), x)
_logt₁(x, df::Float64) = logpdf(TDist(df), x)
_T₁(x, df::Float64) = cdf(TDist(df), x)
_logT₁(x, df::Float64) = logcdf(TDist(df), x)

function _log_g(Q, ν::Float64, k::Int)
    # Adapted from mvtdist_consts in Distributions package
    hdf = 0.5 * ν
    hdim = 0.5 * k
    shdfhdim = hdf + hdim
    v = lgamma(shdfhdim) - lgamma(hdf) - hdim*log(ν) - hdim*log(pi)
    return v - (shdfhdim * log(1 + Q/ν))
end

function scale_and_cor(Ω::Matrix{Float64})
    Ωz = similar(Ω)
    d = size(Ω,1)
    ω = sqrt(diag(Ω))
    for i in 1:d, j in 1:d
        Ωz[i,j] = Ω[i,j]/(ω[i]*ω[j])
    end
    diagm(ω), Ωz
end

function δ(Ω::Matrix{Float64}, α::Vector{Float64})
    ω, Ωz = scale_and_cor(Ω)
    Ωzα = Ωz*α
    αΩzα = dot(α, Ωzα)
    Ωzα/sqrt(1+αΩzα)
end
