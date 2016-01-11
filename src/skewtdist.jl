type SkewTDist <: ContinuousUnivariateDistribution
    ξ::Float64  # Location
    ω::Float64  # Scale
    α::Float64  # Skewness
    df::Float64 # Degrees of freedom
end

function pdf(dist::SkewTDist, x::Real)
    z = (x - dist.ξ)/dist.ω
    2.0 * (_t₁(z, dist.df)/dist.ω) * _T₁(dist.α * z * sqrt((dist.df + 1)/(z^2 + dist.df)), dist.df + 1)
end

function rand(dist::SkewTDist)
    chisqd = Chisq(dist.df)
    w = rand(chisqd)/dist.df
    Ω = Array(Float64, 1,1)
    Ω[1,1] = dist.ω^2
    sndist = MvSkewNormal(Ω, [dist.α])
    x = rand(sndist)[1]
    return dist.ξ + x/sqrt(w)
    ## Ω = Array(Float64, 1,1)
    ## Ω[1,1] = dist.ω^2
    ## return rand(MvSkewTDist([dist.ξ], Ω, [dist.α], dist.df))[1]
end

function cdf(dist::SkewTDist, x::Real)
    quadgk(t->pdf(dist,t), -Inf, x)[1]
end

function quantile(dist::SkewTDist, β::Float64)
    newton(x->cdf(dist,x) - β, dist.ξ)
end

minimum(dist::SkewTDist) = -Inf
maximum(dist::SkewTDist) = Inf

μ(dist::SkewTDist) = δ(dist.α) * sqrt(dist.df/π) * exp(lgamma(0.5*(dist.df-1.0)) - lgamma(0.5*dist.df))
mean(dist::SkewTDist) = dist.ξ + dist.ω * μ(dist)
var(dist::SkewTDist) = (dist.ω^2) * (dist.df/(dist.df - 2.0)) - dist.ω^2 * μ(dist)^2
dof(dist::SkewTDist) = dist.df
