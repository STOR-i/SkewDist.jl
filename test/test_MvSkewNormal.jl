using SkewDist, Distributions, Base.Test

println("Testing MvSkewNormal...")

function cov2cor(X::Matrix{Float64})
    A = similar(X)
    d = size(X,1)
    ωinv = 1.0./sqrt(diag(X))
    for i in 1:d, j in 1:d
        A[i,j] = X[i,j]*ωinv[i]*ωinv[j]
    end
    A
end


srand(1)
N = 100000 # Sample size

ξ = [0.0, -0.1]
α = [1.0, 0.5]
Ω = rand_pd(2)

dist = MvSkewNormal(ξ, Ω, α)
rand(dist)

Z = rand(dist, N)

@test_approx_eq_eps mean(Z, 2) mean(dist) 1e-2
@test_approx_eq_eps var(Z, 2) var(dist) 1e-2
@test_approx_eq_eps Base.covm(Z, mean(Z, 2), 2) cov(dist) 1e-2
