module SkewDist

using Distributions, MultivariateStats, PDMats, Optim
import Base: length, mean, show
import Distributions: _rand!, cov, var, _logpdf, _logpdf!

export MvSkewNormal, MvSkewTDist, fit_MvSkewNormal, fit_MvSkewTDist

include("utils.jl")
include("mvskewnormal.jl")
include("mvskewtdist.jl")
include("fit_mvskewtdist.jl")

end # module
