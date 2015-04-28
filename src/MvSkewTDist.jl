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

t(L::Float64, Q::Float64, ν::Float64, k::Int) = L*sqrt((ν+k)/(Q+ν))
t(L::Vector{Float64}, Q::Vector{Float64}, ν::Float64, k::Int) = L .* sqrt((ν + k)./(Q+ν))

t₁(x::Float64, df::Float64) = pdf(TDist(df), x)
t₁(x::Vector{Float64}, df::Float64) = pdf(TDist(df), x)
T₁(x::Float64, df::Float64) = cdf(TDist(df), x)
T₁(x::Vector{Float64}, df::Float64) = cdf(TDist(df), x)
logT₁(x::Float64, df::Float64) = logcdf(TDist(df), x)
logT₁(x::Vector{Float64}, df::Float64) = logcdf(TDist(df), x)

import Base.LinAlg: ldltfact

function ldltfact(Ω::Matrix{Float64})
    A = chol(Ω)
    d = diag(A)
    scale(1./d, A), d.^2
end

function Q(u::Vector{Float64}, A::Matrix{Float64}, ρ::Vector{Float64})
    # Q = uᵀ Ω⁻¹ u  where Ω⁻¹ = Aᵀ * diag(exp(-2ρ)) * A
    # Now, uᵀΩ⁻¹u = || diag(exp(-ρ)) A u ||²
    # Ωinv = A'diagm(exp(-2*ρ)) * A
    # Qout = dot(u, Ωinv * u)
    return norm(diagm(exp(-ρ)) * A * u)^2
end

function Q(u::Matrix{Float64}, A::Matrix{Float64}, ρ::Vector{Float64})
    n = size(u,1)
    Up = diagm(exp(-ρ)) * A # Upper Cholesky factor
    q = Array(Float64, n)
    for i in 1:n
        q[i] = norm(Up * u[i,:]')^2
    end
    return q
end

L(u::Vector{Float64}, η::Vector{Float64}) = η'u

function log_g(Q::Float64, ν::Float64, k::Int)
    # Adapted from mvtdist_consts in Distributions package
    out=Array(Float64,1)
    try
        hdf = 0.5 * ν
        hdim = 0.5 * k
        shdfhdim = hdf + hdim
        v = lgamma(shdfhdim) - lgamma(hdf) - hdim*log(ν) - hdim*log(pi)
        out[1] = v - (shdfhdim * log(1 + Q/ν))
    catch err
        println("Error: $(err)")
        println("Q = $(Q), ν = $(ν), k = $(k)")
        rethrow(err)
    end
    return out[1]
end

log_g(q::Vector{Float64}, ν::Float64, k::Int) = map(x->log_g(x,ν,k), q)
g_Q(q::Vector{Float64}, ν::Float64, k::Int) = -((ν + k)/2ν)*(1.0./(1 + q))
∂logT₁(t::Float64, ν::Float64, k::Int) = (1.0/T₁(t, ν + k)) * t₁(t, ν+k)
∂logT₁(t::Vector{Float64}, ν::Float64, k::Int) = (1.0./T₁(t, ν + k)) .* t₁(t, ν+k)
t_L(q::Vector{Float64}, ν::Float64, k::Int) = sqrt((ν+k)./(q + ν))
t_q(l::Vector{Float64}, q::Vector{Float64}, ν::Float64, k::Int) = -(l*sqrt(ν+k))./(2*(q + ν).^(1.5))
function ∂log_g∂ν(q::Vector{Float64}, ν::Float64, k::Int)
    0.5 *(digamma(0.5*(ν+k)) - digamma(0.5*ν) - k/ν + ((ν + k)*q)./(ν.^2(1+q/ν)) - log(1+q/ν))
end


function derivative(f)
    return function(x)
        # pick a small value for h
        h = x == 0 ? sqrt(eps(Float64)) : sqrt(eps(Float64)) * x

        # floating point arithmetic gymnastics
        xph = x + h
        dx = xph - x

        # evaluate f at x + h
        f1 = f(xph)

        # evaluate f at x
        f0 = f(x)

        # divide the difference by h
        return (f1 - f0) / dx
    end
end


function ∂logT₁∂ν(l::Vector{Float64}, q::Vector{Float64}, ν::Float64, k::Int)
    lg(ν1::Float64) = logT₁(t(l,q,ν1,k), ν1 + k)
    return derivative(lg)(ν)
end


# Fit a MvSkewTDist regression for model to data
# X = (n x p) model matrix (each row is corresponds to the predictors of one response)
# Y = (n x k) response matrix (each row correspond to one output observation)
#
# The following model is assumed:
#   yᵢ ∼ STₖ(ξᵢ, Ω, α, ν) where ξᵢ = xᵢ̱β
# for some (p x k) design matrix β
#
function fit_MvSkewTDist(X::Matrix{Float64}, Y::Matrix{Float64}; kwargs...)
    # We use the following reparametrization:
    # Ω⁻¹ = Aᵀdiag(exp(-2ρ))A = AᵀDA
    # η = ω⁻¹ α
    # Now, for θ = (β, A, ρ, η, log ν)
    # ℓ(θ) = ∑ᵢ ℓᵢ(θ)
    # for observations i = 1,…,n
    # where
    # ℓᵢ(θ) = log 2 + 0.5 log |D| + log gₖ(Qᵢ; ν) + log T₁( t(Lᵢ, Qᵢ, ν); ν + k )
    # and
    # uᵢ = yᵢ - βᵀ xᵢ
    # Qᵢ = uᵢᵀ Ω⁻¹ uᵢ
    # Lᵢ = αᵀ ω⁻¹ uᵢ
    # and
    # t(L,Q,ν) = L √((ν + k)/(Q + ν))
    # T₁ denots the scalar t₁ distribution function
    # gₖ(Q; ν) = ((Γ((ν + k)/2))/((πν)^(k/2)* Γ(ν/2))) * (1 + Q/ν)^(-(ν + d)/2)
    size(X,1) == size(Y,1) || throw(ArgumentError("X and Y must have the same number of rows"))
    n,p = size(X)
    k = size(Y,2)

    U(β::Matrix{Float64}) = Y - X*β
    
    
    function ll(params::Vector{Float64})
        β = reshape(params[1:(p*k)], p, k)
        A = reshape(params[(p*k + 1):(p*k+k*k)], k, k)
        ρ = params[(p*k+k*k+1):(p*k + k*k + k)]
        η = params[(p*k + k*k + k + 1):(p*k + k*k + 2*k)]
        ν = exp(params[end])

        u = U(β)
        q = Q(u, A, ρ)
        l = u*η 
        lg = log_g(q, ν, k)
        lT = logT₁(t(l, q, ν, k), ν + k)
        
        # logdet(D) = -2 Σᵢρᵢ
        D = diagm(exp(-2ρ))
        ℓ = n * ( log(2) + 0.5 * logdet(D) ) + sum(lg) + sum(lT)
        return ℓ
    end

    function dll!(params::Vector{Float64}, grad::Vector{Float64})
        β = reshape(params[1:(p*k)], p, k)
        A = reshape(params[(p*k + 1):(p*k+k*k)], k, k)
        ρ = params[(p*k+k*k+1):(p*k + k*k + k)]
        η = params[(p*k + k*k + k + 1):(p*k + k*k + 2*k)]
        ν = exp(params[end])

        u = U(β)
        q = Q(u, A, ρ)
        l = u*η 
        lg = log_g(q, ν, k)
        tlqν = t(l, q, ν, k)
        D = diagm(exp(-2*ρ))
        Dinv = diagm(exp(2*ρ))
        Ωinv = A'D*A
        T₁tilde = ∂logT₁(tlqν, ν, k)
        
        
        # Calculate derivatives        
        ∂ℓ∂β = -2X'diagm(g_Q(q, ν, k) + T₁tilde.*t_q(l,q,ν,k))*u*Ωinv - X'diagm(T₁tilde.*t_L(q,ν,k))*ones(n)*η'
        ∂ℓ∂A = 2 * triu( D*A*u'diagm(g_Q(q,ν,k) + T₁tilde .* t_q(l,q,ν,k))*u )
        ∂ℓ∂D = eye(k).*(A*u'diagm(g_Q(q,ν,k) + T₁tilde.*t_q(l,q,ν,k))*u*A') + 0.5 * n * Dinv
        ∂ℓ∂ρ = diag(∂ℓ∂D).*(-2*diag(D))
        
        ∂ℓ∂η = u'diagm(T₁tilde.*t_L(q,ν,k))*ones(n)
        ∂ℓ∂ν = sum( ∂log_g∂ν(q,ν,k) + ∂logT₁∂ν(l,q,ν,k))
        ∂ℓ∂logν = ∂ℓ∂ν*ν
        
        grad[1:(p*k)] = -vec(∂ℓ∂β)
        grad[(p*k + 1):(p*k+k*k)] = -vec(  ∂ℓ∂A)
        grad[(p*k+k*k+1):(p*k + k*k + k)] = -∂ℓ∂ρ
        grad[(p*k + k*k + k + 1):(p*k + k*k + 2*k)] = -∂ℓ∂η
        grad[end] =  -∂ℓ∂logν
       
    end

    function ll_and_dll!(params::Vector{Float64}, grad::Vector{Float64})
        β = reshape(params[1:(p*k)], p, k)
        A = reshape(params[(p*k + 1):(p*k+k*k)], k, k)
        ρ = params[(p*k+k*k+1):(p*k + k*k + k)]
        η = params[(p*k + k*k + k + 1):(p*k + k*k + 2*k)]
        ν = exp(params[end])

        u = U(β)
        q = Q(u, A, ρ)
        l = u*η 
        lg = log_g(q, ν, k)
        tlqν = t(l, q, ν, k)
        D = diagm(exp(-2*ρ))
        Dinv = diagm(exp(2*ρ))
        Ωinv = A'D*A
        T₁tilde = ∂logT₁(tlqν, ν, k)
        
        
        # Calculate derivatives        
        ∂ℓ∂β = -2X'diagm(g_Q(q, ν, k) + T₁tilde.*t_q(l,q,ν,k))*u*Ωinv - X'diagm(T₁tilde.*t_L(q,ν,k))*ones(n)*η'
        ∂ℓ∂A = 2 * triu(D*A*u'diagm(g_Q(q,ν,k) + T₁tilde.*t_q(l,q,ν,k))*u)
        ∂ℓ∂D = eye(k).*(A*u'diagm(g_Q(q,ν,k) + T₁tilde.*t_q(l,q,ν,k))*u*A') + 0.5 * n * Dinv
        ∂ℓ∂ρ = diag(∂ℓ∂D).*(-2*diag(D))
        
        ∂ℓ∂η = u'diagm(T₁tilde.*t_L(q,ν,k))*ones(n)
        ∂ℓ∂ν = sum(∂log_g∂ν(q,ν,k) + ∂logT₁∂ν(l,q,ν,k))
        ∂ℓ∂logν = ∂ℓ∂ν * ν

        grad[1:(p*k)] = -vec(∂ℓ∂β)
        grad[(p*k + 1):(p*k+k*k)] = -vec(  ∂ℓ∂A)
        grad[(p*k+k*k+1):(p*k + k*k + k)] = -∂ℓ∂ρ
        grad[(p*k + k*k + k + 1):(p*k + k*k + 2*k)] = -∂ℓ∂η
        grad[end] =  -∂ℓ∂logν
        
        # logdet(D) = -2 Σᵢρᵢ
        lT = logT₁(t(l, q, ν, k), ν + k)
        ℓ = n * ( log(2) + 0.5 * logdet(D) ) + sum(lg) + sum(lT)
        return ℓ
    end

    func = DifferentiableFunction(ll, dll!, ll_and_dll!)

    init_β = ones(p,k)
    init_A = triu(ones(k,k))
    init_ρ = ones(k)
    init_η = ones(k)
    init_logν = log(4.0)
    init = [vec(init_β), vec(init_A), init_ρ, init_η, init_logν]
    results = optimize(func, init; kwargs...)
    
    β = reshape(results.minimum[1:p*k], p, k)
    A = reshape(results.minimum[p*k+1:p*k+k*k])
    ρ = results.minimum[p*k+k*k + 1:p*k+k*k + k]
    η = results.minimum[p*k+k*k + k + 1:p*k+k*k + 2*k]
    D = diagm(exp(-2*ρ))
    Ωinv = A'D*A
    Ω = inv(Ωinv)
    ω = diagm(sqrt(diag(Ω)))
    α = ω * η

    return MvSkewTDist(zeros(k), Ω, α, df)
end
                   
    
                   
