type SkewTDist <: ContinuousUnivariateDistribution
    ξ::Float64  # Location
    ω::Float64  # Scale
    α::Float64  # Skewness
    df::Float64 # Degrees of freedom
end

skewtdistpdf(α::Real, df::Real, z::Real) = 2.0* _t₁(z, df) * _T₁(α * z * sqrt((df + 1)/(z^2 + df)), df + 1)

function pdf(dist::SkewTDist, x::Real)
    z = (x - dist.ξ)/dist.ω
    skewtdistpdf(dist.α, dist.df, z)/dist.ω
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
    if dist.α < 0
        β = 1 - β
    end
    a = tdistinvcdf(dist.df, β)
    b = sqrt(fdistinvcdf(1.0, dist.df, β))
    qz = fzero(x->cdf(dist, x) - β, a, b)
end

minimum(dist::SkewTDist) = -Inf
maximum(dist::SkewTDist) = Inf

μ(dist::SkewTDist) = δ(dist.α) * sqrt(dist.df/π) * exp(lgamma(0.5*(dist.df-1.0)) - lgamma(0.5*dist.df))
mean(dist::SkewTDist) = dist.ξ + dist.ω * μ(dist)
var(dist::SkewTDist) = (dist.ω^2) * (dist.df/(dist.df - 2.0)) - dist.ω^2 * μ(dist)^2
dof(dist::SkewTDist) = dist.df
