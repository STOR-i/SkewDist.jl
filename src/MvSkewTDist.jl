immutable MvSkewTDist <: Sampleable{Multivariate, Continuous}
    ξ::Vector{Float64}
    Ω::Matrix{Float64}
    α::Vector{Float64}
    df::Float64
end

length(dist::MvSkewTDist) = length(dist.α)

function _rand!{T<:Real}(dist::MvSkewTDist, x::AbstractVector{T})
    chisqd = Chisq(dist.df)
    w = rand(chisqd)/dist.df
    sndist = MvSkewNormal(dist.Ω, dist.α)
    rand!(sndist, x)
    # x = ξ + z/sqrt(x)
    broadcast!(/, x, x, sqrt(w))
    broadcast!(+, x, x, dist.ξ)
end

function δ(dist::MvSkewTDist)
    ω, Ωz = scale_and_cor(dist.Ω)
    Ωzα = Ωz*dist.α
    αΩzα = dot(dist.α, Ωzα)
    Ωzα/sqrt(1+αΩzα)
end

function μ(dist::MvSkewTDist)
    δ(dist) * sqrt(dist.df/π) * (gamma(0.5*(dist.df - 1))/gamma(0.5*dist.df))
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
    (dist.df/(dist.df-2)) * dist.Ω - ω*mu*mu'ω
end



