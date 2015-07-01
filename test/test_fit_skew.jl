using SkewDist

println("Testing fitting functions...")
#srand(1)

n = 5000 # Number of observations
k = 3    # Dimension
p = 4    # Number of covariates

# Randomly generate MvSkewNormal params
α = rand(k)
A = randn(k,k)
Ω = A'A

X = randn(n,p)

β = randn(p, k)
ξ = X*β   # Location parameters

## Z = MvSkewNormal(Ω, α) # MvSkewNormal located at origin
## Y = Array(Float64, n, k)
## for i in 1:n
##     Y[i,:] = rand(Z)' + ξ[i,:]
## end

## fit_MvSkewNormal(X, Y, method=:bfgs, show_trace=true)

## # Test directly for randomly generated data

## ξ = randn(k)
## dist = MvSkewNormal(ξ, Ω, α)
## Y = rand(dist, n)'
## fit_MvSkewNormal(Y)
## fit_MvSkewNormal(Y; method=:bfgs, show_trace=true)


############################

ν = 4.0
dist = MvSkewTDist(zeros(k), Ω, α, ν)

Y = Array(Float64, n, k)
for i in 1:n
    Y[i,:] = rand(dist)' + ξ[i,:]
end

β_fit, Z = fit_MvSkewTDist(X, Y; show_trace=true, method=:bfgs )

ξ = randn(k)
dist2 = MvSkewTDist(ξ, Ω, α, ν)
Y2 = rand(dist2, n)'
dist_fit = fit_MvSkewTDist(Y2; show_trace=true, method=:bfgs)
