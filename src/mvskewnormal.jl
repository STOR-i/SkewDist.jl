immutable MvSkewNormal <: Sampleable{Multivariate, Continuous}
    ξ::Vector{Float64}      # Location vector
    Ω::PDMat                # Covariance matrix
    α::Vector{Float64}      # Shape vector
    # Auxiliary parameters
    δ::Vector{Float64}
    Ωstar::PDMat  # For sampling...
    function MvSkewNormal(Ω::Matrix{Float64}, α::Vector{Float64})
        d = length(α)
        Ωα = Ω*α
        αΩα = dot(α, Ωα)
        δ = Ωα/sqrt(1+αΩα)
        Ωstar = Array(Float64, d+1, d+1)
        Ωstar[:, 1] = [1, δ]
        Ωstar[1,:] = [1 δ']
        Ωstar[2:end, 2:end] = Ω
        Ωstar = PDMat(Ωstar)
        Ω = PDMat(Ω)
        new(Ω, α, δ, Ωstar)
    end
end

length(dist::MvSkewNormal) = length(dist.α)

function _rand!{T<:Real}(dist::MvSkewNormal, out::AbstractVector{T})
    xx = randn(length(out) + 1)
    unwhiten!(dist.Ωstar, xx)
    x0 = xx[1]
    x = xx[2:end]
    for i in 1:length(dist)
        out[i] = (x0 > 0 ? x[i] : -x[i])
    end
end

mean(dist::MvSkewNormal) = sqrt(2/π)*dist.δ
var(dist::MvSkewNormal) = diag(dist.Ω.mat) - mean(dist).^2

function cov(dist::MvSkewNormal)
    μ=mean(dist)
    dist.Ω.mat - μ*μ'
end
