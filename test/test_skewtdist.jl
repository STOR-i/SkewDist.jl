using Base.Test
using SkewDist
println("Testing SkewTDist...")

df = 4.0
ξ = 1.0
ω = 0.5
α = 1.0

dist = SkewTDist(ξ, ω, α, df)

# Test PDF integrates to one
@test_approx_eq_eps quadgk(x->pdf(dist,x), -Inf, Inf)[1] 1.0 1e-6

cdf(dist, 1.0)

# Monte Carlo tests for basic statistics
n = 100000
x = rand(dist, n)
@test_approx_eq_eps mean(x) mean(dist) 1e-1
@test_approx_eq_eps var(x) var(dist) 1e-1

