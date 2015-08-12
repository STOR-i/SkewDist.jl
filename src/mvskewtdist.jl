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

function _rand!(dist::MvSkewTDist, X::DenseMatrix)
    chisqd = Chisq(dist.df)
    w = rand(chisqd, (1,size(X,2)))/dist.df
    sndist = MvSkewNormal(dist.Ω.mat, dist.α)
    rand!(sndist, X)
    broadcast!(/, X, X, sqrt(w))
    broadcast!(+, X, X, dist.ξ)
    X
end

function _logpdf!{T<:Real}(r::AbstractArray, dist::MvSkewTDist, X::DenseMatrix{T})
    k = length(dist)
    Y = broadcast(-, X, dist.ξ)
    Q = invquad(dist.Ω, Y)
    ω = Diagonal(sqrt(diag(dist.Ω)))
    logtd = - 0.5*logdet(dist.Ω) + _log_g(Q, dist.df, k)
    r[:] = log(2) + logtd + _logT₁(dot(dist.α,ω\Y) * sqrt((dist.df + k)./(Q + dist.df)), dist.df + k)
end

function _logpdf{T<:Real}(dist::MvSkewTDist, x::AbstractVector{T})
    k = length(dist)
    Q = invquad(dist.Ω, x - dist.ξ)
    ω = Diagonal(sqrt(diag(dist.Ω)))
    logtd = - 0.5*logdet(dist.Ω) + _log_g(Q, dist.df, k)
    log(2) + logtd + _logT₁(dot(dist.α,ω\(x-dist.ξ)) * sqrt((dist.df + k)/(Q + dist.df)), dist.df + k)
end

function μ(dist::MvSkewTDist)
    δ(dist.Ω, dist.α) * sqrt(dist.df/π) * exp(lgamma(0.5*(dist.df - 1)) - lgamma(0.5*dist.df))
end

function mean(dist::MvSkewTDist)
    ω = Diagonal(sqrt(diag(dist.Ω)))
    dist.ξ + ω * μ(dist)
end

function var(dist::MvSkewTDist)
    ω2 = Diagonal(diag(dist.Ω))
    ω = Diagonal(sqrt(ω2))
    diag(ω2*(dist.df/(dist.df-2))) - (ω*μ(dist)).^2
end

function cov(dist::MvSkewTDist)
    ω = Diagonal(sqrt(diag(dist.Ω)))
    mu = μ(dist)
    (dist.df/(dist.df-2)) * dist.Ω.mat - ω*mu*mu'ω
end
