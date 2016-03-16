_t₁(x, df::Float64) = tdistpdf(df, x)
_t₁(x::Vector, df::Float64) = map(u->tdistpdf(df, u), x)
_logt₁(x, df::Float64) = tdistlogpdf(df, x)
_logt₁(x::Vector, df::Float64) = map(u->tdistlogpdf(df, u), x)
_T₁(x, df::Float64) = tdistcdf(df, x)
_T₁(x::Vector, df::Float64) = map(u->tdistcdf(df, u), x)
_logT₁(x, df::Float64) = tdistlogcdf(df, x)
_logT₁(x::Vector, df::Float64) = map(u->tdistlogcdf(df, u), x)

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

function scale_and_cor(Ω::PDMat)
    Ωz = similar(Ω.mat)
    d = size(Ω,1)
    ω = sqrt(diag(Ω))
    for i in 1:d, j in 1:d
        Ωz[i,j] = Ω.mat[i,j]/(ω[i]*ω[j])
    end
    Diagonal(ω), PDMat(Ωz)
end

δ(α::Float64) = α/sqrt(1.0 + α^2)

function δ(Ω::Matrix{Float64}, α::Vector{Float64})
    ω, Ωz = scale_and_cor(Ω)
    Ωzα = Ωz*α
    αΩzα = dot(α, Ωzα)
    Ωzα/sqrt(1+αΩzα)
end

function δ(Ω::PDMat, α::Vector{Float64})
    ω, Ωz = scale_and_cor(Ω)
    (Ωz * α)/sqrt(1.0 + quad(Ωz, α))
end
