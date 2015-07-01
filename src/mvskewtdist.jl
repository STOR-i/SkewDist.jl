immutable MvSkewTDist <: ContinuousMultivariateDistribution
    ξ::Vector{Float64}
    Ω::PDMat
    α::Vector{Float64}
    df::Float64
end

function MvSkewTDist(ξ::Vector{Float64}, Ω::Matrix{Float64}, α::Vector{Float64}, df::Float64)
    MvSkewTDist(ξ, PDMat(Ω), α, df)
end
    
length(dist::MvSkewTDist) = length(dist.α)

length(dist::MvSkewTDist) = length(dist.α)

function _rand!{T<:Real}(dist::MvSkewTDist, x::AbstractVector{T})
    chisqd = Chisq(dist.df)
    w = rand(chisqd)/dist.df
    sndist = MvSkewNormal(dist.Ω.mat, dist.α)
    rand!(sndist, x)
    # x = ξ + z/sqrt(x)
    broadcast!(/, x, x, sqrt(w))
    broadcast!(+, x, x, dist.ξ)
    x
end

function _logpdf!{T<:Real}(r::AbstractArray, dist::MvSkewTDist, X::DenseMatrix{T})
    k = length(dist)
    Y = broadcast(-, X, dist.ξ)
    Q = invquad(dist.Ω, Y)
    ωinv = diagm(sqrt(1./diag(dist.Ω)))
    logtd = - 0.5*logdet(dist.Ω) + _log_g(Q, dist.df, k)
    r[:] = log(2) + logtd + _logT₁(dot(dist.α,ωinv*Y) * sqrt((dist.df + k)./(Q + dist.df)), dist.df + k)
end

function _logpdf{T<:Real}(dist::MvSkewTDist, x::AbstractVector{T})
    k = length(dist)
    Q = invquad(dist.Ω, x - dist.ξ)
    ωinv = diagm(sqrt(1./diag(dist.Ω)))
    logtd = - 0.5*logdet(dist.Ω) + _log_g(Q, dist.df, k)
    log(2) + logtd + _logT₁(dot(dist.α,ωinv*(x-dist.ξ)) * sqrt((dist.df + k)/(Q + dist.df)), dist.df + k)
end

function μ(dist::MvSkewTDist)
    δ(dist.Ω.mat, dist.α) * sqrt(dist.df/π) * (gamma(0.5*(dist.df - 1))/gamma(0.5*dist.df))
end

function mean(dist::MvSkewTDist)
    ω = diagm(sqrt(diag(dist.Ω)))
    dist.ξ + ω * μ(dist)
end

function var(dist::MvSkewTDist)
    ω = diagm(sqrt(diag(dist.Ω)))
    ω2 = diagm(diag(dist.Ω))
    diag(ω2*(dist.df/(dist.df-2))) - (ω*μ(dist)).^2
end

function cov(dist::MvSkewTDist)
    ω = diagm(sqrt(diag(dist.Ω)))
    mu = μ(dist)
    (dist.df/(dist.df-2)) * dist.Ω.mat - ω*mu*mu'ω
end
