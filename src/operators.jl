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
