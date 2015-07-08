using SkewDist, Distributions, Base.Test

function rand_pd(d::Int)
    A = randn(d,d)
    A'A
end

println("Testing operators...")

srand(11)

N = 1000000

d = 3
ξ = rand(d)
α = rand(d)
Ω = rand_pd(d)
ν = 4.0
A = rand(d-1,d)

dist = MvSkewTDist(ξ, Ω, α, 4.0)
println(dist)
trans_dist = A * dist
println(trans_dist)
x = A * rand(dist, N)
println(x[:,1:5])

println("Means...")
println(mean(x,2))
println(mean(trans_dist))
@test_approx_eq_eps mean(x, 2) mean(trans_dist) 1e-1

println("Variances...")
println(var(x,2))
println(var(trans_dist))
@test_approx_eq_eps var(x,2) var(trans_dist) 1e-1
@test_approx_eq_eps cov(x, vardim=2) cov(trans_dist) 1e-1
