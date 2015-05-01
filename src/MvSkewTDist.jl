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



_t₁(x::Vector{Float64}, df::Float64) = pdf(TDist(df), x)
_logt₁(x::Vector{Float64}, df::Float64) = logpdf(TDist(df), x)
_T₁(x::Vector{Float64}, df::Float64) = cdf(TDist(df), x)
_logT₁(x::Vector{Float64}, df::Float64) = logcdf(TDist(df), x)

import Base.LinAlg: ldltfact

function ldltfact(Ω::Matrix{Float64})
    A = chol(Ω)
    d = diag(A)
    scale(1./d, A), d.^2
end

function _Q(U::Matrix{Float64}, A::Matrix{Float64}, ρ::Vector{Float64})
    n = size(U,1)
    Up = diagm(exp(-ρ)) * A # Upper Cholesky factor
    q = Array(Float64, n)
    for i in 1:n
        q[i] = norm(Up * U[i,:]')^2
    end
    return q
end

_L(U::Matrix{Float64}, η::Vector{Float64}) = U*η

function _log_g(Q::Vector{Float64}, ν::Float64, k::Int)
    # Adapted from mvtdist_consts in Distributions package
    hdf = 0.5 * ν
    hdim = 0.5 * k
    shdfhdim = hdf + hdim
    v = lgamma(shdfhdim) - lgamma(hdf) - hdim*log(ν) - hdim*log(pi)
    return v - (shdfhdim * log(1 + Q./ν))
end

_t(L::Vector{Float64}, Q::Vector{Float64}, ν::Float64, k::Int) = L .* sqrt((ν + k)./(Q+ν))
_sf(Q::Vector{Float64}, ν::Float64, k::Int) = sqrt((ν+k)./(ν+Q))
#_sf2(Q::Vector{Float64}, ν::Float64, k::Int) = sqrt((1+k/ν)./(1+(Q./ν))) # R uses this definition for large ν
_gQ(sf::Vector{Float64}) = (-0.5)*sf.^2 #-((ν + k)/2ν)*(1.0./(1 + q))
_∂logT₁∂t(t::Vector{Float64}, ν::Float64, k::Int) = exp(_logt₁(t, ν+k) - _logT₁(t,ν+k))
_tL(sf::Vector{Float64}) = sf
_tQ(l::Vector{Float64}, q::Vector{Float64}, sf::Vector{Float64}, ν::Float64) = (-0.5).*l.*sf./(q+ν) 
function _∂log_g∂ν(q::Vector{Float64}, ν::Float64, k::Int)
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


function _∂logT₁∂ν(l::Vector{Float64}, q::Vector{Float64}, ν::Float64, k::Int)
    lg(ν1::Float64) = _logT₁(_t(l,q,ν1,k), ν1 + k)
    return derivative(lg)(ν)
end

function read_params(params::Vector{Float64}, p::Int, k::Int)
    β = reshape(params[1:(p*k)], p, k)
    ρ = params[(p*k+1):(p*k + k)]
    i0 = p*k + k
    A = eye(k)
    for j in 1:k-1
        for i in 1:j
            A[i,j+1] = params[i0 + ((j-1)*j)/2 + i]
        end
    end
    i1 = p*k + (k*(k+1))/2
    η = params[i1 + 1:i1 + k]
    ν = exp(params[end])
    return (β, ρ, A, η, ν)
end

function write_params(β::Matrix{Float64}, ρ::Vector{Float64}, A::Matrix{Float64}, η::Vector{Float64}, ν::Float64)
    p, k = size(β)
    params = Array(Float64, p*k + div((k+2)*(k+1),2))
    params[1:p*k] = vec(β)
    params[p*k + 1: p*k + k] = ρ
    i0 = p*k + k
    for j in 1:k-1
        for i in 1:j
            params[i0 + ((j-1)*j)/2 + i] = A[i,j+1]
        end
    end
    i1 = p*k + (k*(k+1))/2
    params[i1 + 1:i1 + k] = η
    params[end] = ν
    return params
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

    _U(β::Matrix{Float64}) = Y - X*β
    
    
    function ll(params::Vector{Float64})
        (β, ρ, A, η, ν) = read_params(params, p, k)
        
        U = _U(β)
        Q = _Q(U, A, ρ)
        L = _L(U,η)
        t = _t(L, Q, ν, k)
        
        # logdet(D) = -2 Σᵢρᵢ
        D = diagm(exp(-2ρ))
        ℓ = n * ( log(2) + 0.5 * logdet(D) ) + sum( _log_g(Q, ν, k) + _logT₁(t, ν + k) )

        return -ℓ
    end

    function dll!(params::Vector{Float64}, grad::Vector{Float64})
        (β, ρ, A, η, ν) = read_params(params, p, k)

        U = _U(β)
        Q = _Q(U, A, ρ)
        L = _L(U,η)
        t = _t(L, Q, ν, k)
        
        D = diagm(exp(-2*ρ))
        Dinv = diagm(exp(2*ρ))
        Ωinv = A'D*A
        sf = _sf(Q,ν,k)
        ∂logT₁∂t = _∂logT₁∂t(t, ν, k)
        g_Q = _gQ(sf)
        t_Q = _tQ(L,Q,sf,ν)
        
        # Calculate derivatives        
        ∂ℓ∂β = -2X'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*Ωinv - X'diagm(∂logT₁∂t.*_tL(sf))*ones(n)*η'
        ∂ℓ∂A = 2 * triu(D*A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U)
        ∂ℓ∂D = eye(k).*(A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*A') + 0.5 * n * Dinv
        ∂ℓ∂ρ = diag(∂ℓ∂D).*(-2*diag(D))
        
        ∂ℓ∂η = U'diagm(∂logT₁∂t.*_tL(sf))*ones(n)
        ∂ℓ∂ν = sum(_∂log_g∂ν(Q,ν,k) + _∂logT₁∂ν(L,Q,ν,k))
        ∂ℓ∂logν = ∂ℓ∂ν * ν

        grad[1:(p*k)] = -vec(∂ℓ∂β)
        grad[(p*k + 1):(p*k+k*k)] = -vec(  ∂ℓ∂A)
        grad[(p*k+k*k+1):(p*k + k*k + k)] = -∂ℓ∂ρ
        grad[(p*k + k*k + k + 1):(p*k + k*k + 2*k)] = -∂ℓ∂η
        grad[end] =  -∂ℓ∂logν
    end

    function ll_and_dll!(params::Vector{Float64}, grad::Vector{Float64})
        (β, ρ, A, η, ν) = read_params(params, p, k)

        U = _U(β)
        Q = _Q(U, A, ρ)
        L = _L(U,η)
        t = _t(L, Q, ν, k)
        
        D = diagm(exp(-2*ρ))
        Dinv = diagm(exp(2*ρ))
        Ωinv = A'D*A
        sf = _sf(Q,ν,k)
        ∂logT₁∂t = _∂logT₁∂t(t, ν, k)
        g_Q = _gQ(sf)
        t_Q = _tQ(L,Q,sf,ν)
        
        # Calculate derivatives        
        ∂ℓ∂β = -2X'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*Ωinv - X'diagm(∂logT₁∂t.*_tL(sf))*ones(n)*η'
        ∂ℓ∂D = eye(k).*(A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*A') + 0.5 * n * Dinv
        ∂ℓ∂ρ = diag(∂ℓ∂D).*(-2*diag(D))

        ∂ℓ∂A = 2 * triu(D*A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U)
        
        ∂ℓ∂η = U'diagm(∂logT₁∂t.*_tL(sf))*ones(n)
        ∂ℓ∂ν = sum(_∂log_g∂ν(Q,ν,k) + _∂logT₁∂ν(L,Q,ν,k))
        ∂ℓ∂logν = ∂ℓ∂ν * ν

        grad[:] = -write_params(∂ℓ∂β, ∂ℓ∂ρ, ∂ℓ∂A, ∂ℓ∂η, ∂ℓ∂logν)
        
        # logdet(D) = -2 Σᵢρᵢ
        ℓ = n * ( log(2) + 0.5 * logdet(D) ) + sum(_log_g(Q, ν, k)) + sum(_logT₁(t, ν + k))
        #println(ℓ)
        return -ℓ
    end

    func = DifferentiableFunction(ll, dll!, ll_and_dll!)
    init_β = ones(p,k)
    init_A = triu(ones(k,k))
    init_ρ = ones(k)
    init_η = ones(k)
    init_ν = 4.0
    params = write_params(init_β, init_ρ, init_A, init_η, init_ν)
    results = optimize(func, params; kwargs...)
    print(results)
    (β, ρ, A, η, ν) = read_params(results.minimum, p, k)
    D = diagm(exp(-2*ρ))
    Ωinv = A'D*A
    Ω = inv(Ωinv)
    ω = diagm(sqrt(diag(Ω)))
    α = ω * η

    return β, MvSkewTDist(zeros(k), Ω, α, ν)
end
                   
    
                   
