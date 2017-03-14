using SkewDist
using Base.Test

function rand_pd(d::Int)
    A = randn(d,d)
    A'A
end

function rand_cor(d::Int)
    A = randn(d,d)
    Σ = A'A
    cov2cor(Σ)
end


include("test_MvSkewNormal.jl")
include("test_MvSkewTDist.jl")
include("test_operators.jl")
include("test_fit_skew.jl")
include("test_skewtdist.jl")
