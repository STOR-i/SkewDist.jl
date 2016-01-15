type SkewNormalSampler <: Sampleable{Univariate,Continuous}
    ξ::Float64
    ω::Float64
    aux::MvNormal
    function SkewNormalSampler(ξ::Float64, ω::Float64, α::Float64)
        δ = α/(1 + α^2)
        aux = MvNormal([[1 δ]; [δ 1]])
        new(ξ, ω, aux)
    end
end

function rand(dist::SkewNormalSampler)
    x0, x = rand(dist.aux)
    dist.ω * ((x0 > 0) ? x : -x) + dist.ξ
end

type SkewNormal <: ContinuousUnivariateDistribution
    ξ::Float64  # Location
    ω::Float64  # Scale
    α::Float64  # Skewness
end

sampler(dist::SkewNormal) = SkewNormalSampler(dist.ξ, dist.ω, dist.α)

function pdf(dist::SkewNormal, x::Real)
    z = (x - dist.ξ)/dist.ω
    2.0 * (normpdf(z)/dist.ω) * normcdf(dist.α * z)
end

function cdf(dist::SkewNormal, x::Real)
    quadgk(t->pdf(dist,t), -Inf, x)[1]
end

function quantile(dist::SkewNormal, β::Float64)
    newton(x->cdf(dist,x) - β, dist.ξ)
end

minimum(dist::SkewNormal) = -Inf
maximum(dist::SkewNormal) = Inf

mean(dist::SkewNormal) = dist.ω*sqrt(2/π)*(α/(1.0 + dist.α^2)) + dist.ξ
var(dist::SkewNormal) = (dist.ω^2) * (1.0 - (sqrt(2/π)*(α/(1.0 + dist.α^2)))^2)

# Cumulant generating function

cgf(dist::SkewNormal, t::Real) = t*dist.ξ + 0.5*t^2*dist.ω^2 + log(2normcdf((dist.ω*t)/(1 + α^2)))
mgf(dist::SkewNormal, t::Real) = exp(cgf(dist, t))
