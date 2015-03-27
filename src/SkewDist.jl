module SkewDist

using Distributions, PDMats
import Base: length, mean
import Distributions: _rand!, cov, var

export MvSkewNormal

include("mvskewnormal.jl")

end # module
