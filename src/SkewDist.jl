module SkewDist

using Distributions, PDMats, Optim
import Base: length, mean, show
import Distributions: _rand!, cov, var

export MvSkewNormal, fit_skew

include("mvskewnormal.jl")

end # module
