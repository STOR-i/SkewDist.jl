n = 1000 # Number of observations
k = 3    # Dimension
p = 4    # Number of covariates

# Randomly generate MvSkewNormal params
α = rand(k)
A = randn(k,k)
Ω = A'A

X = randn(n,p)
β = randn(p, k)
ξ = X*β   # Location parameters

Z = MvSkewNormal(Ω, α) # MvSkewNormal located at origin
Y = Array(Float64, n, k)
for i in 1:n
    Y[i,:] = rand(Z)' + ξ[i,:]
end

fit_skew(X, Y)
