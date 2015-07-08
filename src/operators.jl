function *(A::Matrix{Float64}, dist::MvSkewTDist)
    size(A, 2) == length(dist) || throw(ArgumentError("Matrix and distribution must have compatible dimensions"))
    ξ = A*dist.ξ
    Ω = PDMat(X_A_Xt(dist.Ω, A))
    distω, distΩz = scale_and_cor(dist.Ω)
    ω, Ωz =  scale_and_cor(Ω)
    B = distω\(dist.Ω*A')
    c = sqrt(1 + dot(dist.α,(distΩz.mat - X_invA_Xt(Ω, B))*dist.α))
    α = (ω * (Ω\(B'dist.α)))/c
    return MvSkewTDist(ξ, Ω, α, dist.df)
end

function *(x::Vector{Float64}, dist::MvSkewTDist)
    length(x) == length(dist) || throw(ArgumentError("Vector and distribution must have compatible dimensions"))    
    ξ = dot(x, dist.ξ)
    ω = sqrt(quad(dist.Ω, x))
    distω, distΩz = scale_and_cor(dist.Ω)
    b = distω\(dist.Ω*x)
    c = sqrt(1.0 + dot(dist.α, (distΩz.mat - (b*b')/ω^2)*dist.α))
    α =  dot(b,dist.α)/(c*ω)
    SkewTDist(ξ, ω, α, dist.df)
end

function marginals(dist::MvSkewTDist, indices::Vector{Int})
    n = length(dist)
    (all(0 .< indices) && all(indices .<= n)) || throw(ArgumentError("Indices must be between 1 and $(n)"))
    A = zeros(Float64, length(indices), n)
    for (i, k) in enumerate(indices)
        A[i,k] = 1.0
    end
    return A*dist
end
    
function marginals(dist::MvSkewTDist, i::Int)
    n = length(dist)
    (i > 0 && i < n) || throw(ArgumentError("Indices must be between 1 and $(n)"))
    x = zeros(n)
    x[i] = 1.0
    return x * dist
end
