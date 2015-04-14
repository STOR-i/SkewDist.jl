# See "Statistical Applications of the multivariate skew-normal distribution" by Azzalini and Capitanio (1998)
# for notations and explanations

using Distributions: φ, Φ

function scale_and_cor(Ω::Matrix{Float64})
    Ωz = similar(Ω)
    d = size(Ω,1)
    ω = sqrt(diag(Ω))
    for i in 1:d, j in 1:d
        Ωz[i,j] = Ω[i,j]/(ω[i]*ω[j])
    end
    diagm(ω), Ωz
end

immutable MvSkewNormal <: Sampleable{Multivariate, Continuous}
    ξ::Vector{Float64}      # Location vector
    ω::Matrix{Float64}      # Scale (diagonal matrix)
    Ωz::PDMat               # Correlation matrix
    α::Vector{Float64}      # Shape vector
    # Auxiliary parameters
    δ::Vector{Float64}
    Ωstar::PDMat  # For sampling...
    function MvSkewNormal(ξ::Vector{Float64}, Ω::Matrix{Float64}, α::Vector{Float64})
        ω, Ωz = scale_and_cor(Ω)
        d = length(α)
        Ωzα = Ωz*α
        αΩzα = dot(α, Ωzα)
        δ = Ωzα/sqrt(1+αΩzα)
        Ωstar = Array(Float64, d+1, d+1)
        Ωstar[:, 1] = [1, δ]
        Ωstar[1,:] = [1 δ']
        Ωstar[2:end, 2:end] = Ωz
        Ωstar = PDMat(Ωstar)
        Ωz = PDMat(Ωz)
        new(ξ, ω, Ωz, α, δ, Ωstar)
    end
end

length(dist::MvSkewNormal) = length(dist.α)

function _rand!{T<:Real}(dist::MvSkewNormal, out::AbstractVector{T})
    xx = randn(length(out) + 1)
    unwhiten!(dist.Ωstar, xx)
    x0 = xx[1]
    x = xx[2:end]
    copy!(out, dist.ω * (x0 > 0 ? x : -x) + dist.ξ)
end

mean(dist::MvSkewNormal) = sqrt(2/π)*dist.ω*dist.δ + dist.ξ
var(dist::MvSkewNormal) = (dist.ω.^2)*(diag(dist.Ωz.mat) - (sqrt(2/π)*dist.δ).^2)

function cov(dist::MvSkewNormal)
    μz=sqrt(2/π)*dist.δ
    dist.ω * (dist.Ωz.mat - μz*μz') * dist.ω
end

# Special functions (related to cumulant function of half-normal distribution)
ζ₀(x::Float64) = log(2π * Φ(x))
ζ₀(x::Vector{Float64}) = map(ζ₀, x)

ζ₁(x::Float64) = φ(x)/Φ(x)
ζ₁(x::Vector{Float64}) = map(ζ₁, x)

# Fit a MvSkewNormal regression model for model to data
# X = (n x p) model matrix (each row is corresponds to the predictors of one response)
# Y = (n x k) response matrix (each row correspond to one output observation)
#
# The following model is assumed:
#   yᵢ ∼ SNₖ(ξᵢ, Ω, α) where ξᵢ = xᵢ̱β
# for some (p x k) matrix β of parameters
function fit_skew(X::Matrix{Float64}, Y::Matrix{Float64})
    size(X,1) == size(Y,1) || throw(ArgumentError("X and Y must have the same number of rows"))
    n,p = size(X)
    k = size(Y,2)
    # β is a (p x k) parameter matrix
    u(β::Matrix{Float64}) = Y - X*β
    V(β::Matrix{Float64}) = (u(β)'u(β))/n

    function ll(params::Vector{Float64})
        β = reshape(params[1:(p*k)], p, k)
        η = params[p*k+1:end]
        -0.5 * n * log(det(V(β))) - 0.5 * n * k + sum(ζ₀(u(β)*η))
    end
    
    function dll!(params::Vector{Float64}, grad::Vector{Float64})
        β = reshape(params[1:(p*k)], p, k)
        η = params[p*k+1:end]
        
         ∂l∂β = X'u(β)*inv(V(β)) - X'ζ₁(u(β)*η)*η'
        
        ∂l∂η= u(β)'ζ₁(u*η)
        grad[1:p*k] = vec(∂l∂β)
        grad[p*k + 1:end] = ∂l∂η
    end

    function ll_and_dll!(params::Vector{Float64}, grad::Vector{Float64})
        β = reshape(params[1:p*k], p, k)
        η = params[p*k+1:end]

         ∂l∂β = X'u(β)*inv(V(β)) - X'ζ₁(u(β)*η)*η'
        ∂l∂η = u(β)'ζ₁(u*η)
        grad[1:p*k] = vec(∂l∂β)
        grad[p*k + 1:end] = ∂l∂η
        
        -0.5 * n * log(det(V(β))) - 0.5 * n * k + sum(ζ₀(u(β)*η))
    end
    
end    
