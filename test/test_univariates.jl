using Distributions
using Base.Test
using SkewDist

if isa(Pkg.installed("Calculus"), VersionNumber)
    using Calculus
    function mean_mgf_consistent(dist::ContinuousUnivariateDistribution)
        d = derivative(x->mgf(dist, x))
        m0 = d(0)
        @test_approx_eq derivative(x->mgf(dist, x), 0.0) mean(dist)
    end
    function std_mgf_consistent(dist::ContinuousUnivariateDistribution)
        d1 = derivative(x->mgf(dist,x))
        d2 = derivative(d1)
        @test_approx_eq_eps sqrt(d2(0) - d1(0)^2)  std(dist) 1e-6
    end
else
    mean_mgf_consistent(dist::ContinuousUnivariateDistribution) = true
    std_mgf_consistent(dist::ContinuousUnivariateDistribution) = true
end


function pdf_integrates_to_one(dist::ContinuousUnivariateDistribution)
    @test_approx_eq quadgk(t->pdf(dist, t), minimum(dist), maximum(dist))[1] 1.0
end

function mean_pdf_consistent(dist::ContinuousUnivariateDistribution)
    @test_approx_eq quadgk(t->t*pdf(dist, t), minimum(dist), maximum(dist))[1] mean(dist)
end

function quantile_cdf_consistent(dist::ContinuousUnivariateDistribution)
    for β in 0.05:0.05:0.95
        @test_approx_eq β cdf(dist, quantile(dist, β))
    end
end

# function mean_rand_consistent(dist::ContinuousUnivariateDistribution, n::Int, β::Real)
#     samp = rand(dist, n)
#     m0 = mean(samp)
    
# end

function test_dist(dist::ContinuousUnivariateDistribution)
    println("Testing distribution $(dist)...")
    pdf_integrates_to_one(dist)
    mean_pdf_consistent(dist)
    quantile_cdf_consistent(dist)
    try
        mean_mgf_consistent(dist)
        std_mgf_consistent(dist)
    catch err
        if !isa(err, MethodError)
            rethrow(err)
        end
    end
end

skewnorm1 = SkewNormal(1.0, 1.0, -1.0)
test_dist(skewnorm1)

skewtdist1 = SkewTDist(1.0, 2.0, 2.0, 3.0)
test_dist(skewtdist1)


