immutable MvSkewTDist <: Sampleable{Multivariate, Continuous}
    ξ::Vector{Float64}
    Σ::PDMat
    α::Vector{Float64}
    df::Float64
end

length(dist::MvSkewTDist) = length(dist.α)

function _rand!{T<:Real}(dist::MvSkewTDist, x::AbstractVector{T})
    d = length(dist)
    chisqd = Chisq(d.df)
    x = rand(chisqd, df)/df
    sndist = MvSkewNormal(zeros(d), d.Σ, d.α)
    z = rand(sndist)
    # x = ξ + z/sqrt(x)
    broadcast!(/, x, z, sqrt(x))
    broadcast!(+, x, x, dist.ξ)
end

