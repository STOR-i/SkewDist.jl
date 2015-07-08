using SkewDist, Distributions, Base.Test

function rand_pd(d::Int)
    A = randn(d,d)
    A'A
end

println("Testing operators...")

srand(2)

N = 100000

d = 3
ξ = rand(d)
α = rand(d)
Ω = rand_pd(d)/(2*d)
ν = 4.0
A = rand(d-1,d)
x = rand(d)
dist = MvSkewTDist(ξ, Ω, α, 4.0)

println("\tMatrix transformation...")
dist2 = A * dist
y = A * rand(dist, N)
@test_approx_eq_eps mean(y, 2) mean(dist2) 1e-1
@test_approx_eq_eps var(y,2) var(dist2) 1e-1
@test_approx_eq_eps cov(y, vardim=2) cov(dist2) 1e-1

println("\tLinear combination...")
dist3 = x * dist
z = x'rand(dist, N)
@test_approx_eq_eps mean(z) mean(dist3) 1e-2
@test_approx_eq_eps var(z) var(dist3) 1e-2

println("\tLinear and matrix transformations consistent...")
dist4 = x' * dist
@test_approx_eq dist4.ξ[1] dist3.ξ
@test_approx_eq sqrt(dist4.Ω.mat[1,1]) dist3.ω
@test_approx_eq dist4.α[1] dist3.α
