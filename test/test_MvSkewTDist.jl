using SkewDist, Distributions, Base.Test

println("Testing MvSkewTDist...")

function rand_pd(d::Int)
    A = randn(d,d)
    A'A
end

srand(1)
N = 1000000 # Sample size

ξ = [0.0, -0.1]
α = [1.0, 0.5]
Ω = rand_pd(2)
df = 4.0

dist = MvSkewTDist(ξ, Ω, α, df)
rand(dist)

Z = rand(dist, N)

@test_approx_eq_eps mean(Z, 2) mean(dist) 5e-3
@test_approx_eq_eps var(Z, 2) var(dist) 1e-2
@test_approx_eq_eps cov(Z, vardim=2) cov(dist) 1e-2
