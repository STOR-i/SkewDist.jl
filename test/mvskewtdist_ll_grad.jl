# This script steps through the calculation of the gradient of the log-likelihood
# for the multivariate skew t distribution. For debugging purposes.

using SkewDist

n, p = size(X)
k = size(Y,2)
(β, ρ, A, η, ν) = SkewDist.read_params(params, p, k)
# print_params(β, ρ, A, η, ν)
# println("————————————–")
U = SkewDist._U(X,Y,β)
Q = SkewDist._Q(U, A, ρ)
L = SkewDist._L(U,η)
t = SkewDist._t(L, Q, ν, k)
    
D = diagm(exp(-2*ρ))
Dinv = diagm(exp(2*ρ))
Ωinv = A'D*A
sf = SkewDist._sf(Q,ν,k)
∂logT₁∂t = SkewDist._∂logT₁∂t(t, ν, k)
g_Q = SkewDist._gQ(sf)
t_Q = SkewDist._tQ(L,Q,sf,ν)
    
# Calculate derivatives        
∂ℓ∂β = -2X'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*Ωinv - X'diagm(∂logT₁∂t.*SkewDist._tL(sf))*ones(n)*η'
∂ℓ∂D = eye(k).*(A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U*A') + 0.5 * n * Dinv
∂ℓ∂ρ = diag(∂ℓ∂D).*(-2*diag(D))

∂ℓ∂A = 2 * triu(D*A*U'diagm(g_Q + ∂logT₁∂t.*t_Q)*U)

∂ℓ∂η = U'diagm(∂logT₁∂t.*SkewDist._tL(sf))*ones(n)
∂ℓ∂ν = sum(SkewDist._∂log_g∂ν(Q,ν,k) + SkewDist._∂logT₁∂ν(L,Q,ν,k))
∂ℓ∂logν = ∂ℓ∂ν *

#print_params(∂ℓ∂β, ∂ℓ∂ρ, ∂ℓ∂A , ∂ℓ∂η, ∂ℓ∂logν)
ℓ = n * ( log(2) + 0.5 * logdet(D) ) + sum(SkewDist._log_g(Q, ν, k)) + sum(SkewDist._logT₁(t, ν + k))

grad = -2*SkewDist.write_params(∂ℓ∂β, ∂ℓ∂ρ, ∂ℓ∂A, ∂ℓ∂η, ∂ℓ∂logν)
grad
