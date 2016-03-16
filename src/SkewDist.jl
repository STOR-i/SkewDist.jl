module SkewDist

using Distributions, StatsFuns, MultivariateStats, PDMats, Optim, Roots
import Base: length, mean, show, rand, var, cov, *
import Distributions: _rand!, pdf, _logpdf, _logpdf!, dof, cdf, quantile, minimum, maximum

export MvSkewNormal, MvSkewTDist, SkewNormal, SkewTDist, fit_MvSkewNormal, fit_MvSkewTDist, marginals, mgf
export pdf, dof, cdf, quantile

include("utils.jl")
include("mvskewnormal.jl")
include("mvskewtdist.jl")
include("skewnormal.jl")
include("skewtdist.jl")
include("operators.jl")
include("fit_mvskewtdist.jl")

end # module
