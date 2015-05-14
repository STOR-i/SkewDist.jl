module SkewDist

using Distributions, PDMats, Optim
import Base: length, mean, show
import Distributions: _rand!, cov, var

export MvSkewNormal, MvSkewTDist, fit_MvSkewNormal, fit_MvSkewTDist

include("utils.jl")
include("mvskewnormal.jl")
include("MvSkewTDist.jl")
include("fit_mvskewtdist.jl")

end # module
