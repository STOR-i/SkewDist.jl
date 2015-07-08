module SkewDist

using Distributions, MultivariateStats, PDMats, Optim
import Base: length, mean, show, rand, var, cov
import Distributions: _rand!, pdf, _logpdf, _logpdf!, dof

export MvSkewNormal, MvSkewTDist, SkewTDist, fit_MvSkewNormal, fit_MvSkewTDist
export pdf, dof

include("utils.jl")
include("mvskewnormal.jl")
include("mvskewtdist.jl")
include("skewtdist.jl")
include("operators.jl")
include("fit_mvskewtdist.jl")

end # module
