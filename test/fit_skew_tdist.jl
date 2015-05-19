# Script used to calculate log-likelihood of multivariate skew t distribution
# for comparison with Julia code. Note that the R script of the same name must be run
# before this, as this script uses its output.

using MultivariateStats, SkewDist

# Load R data
X = readdlm("x.txt", ' ')
Y = readdlm("y.txt", ' ')

n, k = size(Y)
p = size(X,2)

βinit = readdlm("Beta.txt", ' ')
Ωinit = readdlm("Omega.txt", ' ')
αinit = ones(k)
νinit = 4.0

ρ, A, η, logν = SkewDist.dplist2optpar(Ωinit, αinit, νinit)
params = SkewDist.write_params(βinit, ρ, A, η, logν)



println("Parameter vector:", params)
nll, grad = SkewDist.nll_and_grad(params, X, Y)
println("Neg. Log Likelihood: ", nll)
print("Gradient: ", grad, "\n")

βinit = llsq(X,Y; bias=false)
resid = Y - X*βinit
