# Script used to calculate log-likelihood of multivariate skew t distribution
# for comparison with Julia code

library("sn")
library("mnormt")

mst.pdev <- function(param, x, y, w, fixed.nu=NULL, symmetr=FALSE, 
   penalty=NULL, trace=FALSE)
{
  if(missing(w)) w <- rep(1,nrow(y))
  d <- ncol(y)
  p <- ncol(x)
  npar0 <- (p*d + d*(d+1)/2)
  param1 <- c(param[1:npar0], if(symmetr) rep(0, d) else param[npar0+(1:d)], 
    if(is.null(fixed.nu)) param[length(param)])
  dp.list <- optpar2dplist(param1, d, p)
  #browser()
  dp <- dp.list$dp
  nu <- if(is.null(fixed.nu)) dp$nu else fixed.nu
  logL <- sum(w * dmst(y, x %*% dp$beta, dp$Omega, dp$alpha, nu, log=TRUE))
  Q <- if(is.null(penalty)) 0 else 
       penalty(list(alpha=dp$alpha, Omega.bar=cov2cor(dp$Omega)), nu, der=0)
  pdev <- (-2) * (logL - Q)
  if(trace) cat("mst.pdev: ", pdev, "\nparam:", format(param), "\n")
  pdev
}

optpar2dplist <- function(param, d, p, x.names=NULL, y.names=NULL)
{# convert vector form of optimization parameters to DP list;
 # output includes inverse(Omega) and its log determinant 
  beta <- matrix(param[1:(p * d)], p, d)
  D <- exp(-2 * param[(p * d + 1):(p * d + d)])
  A <- diag(d)
  i0 <- p*d + d*(d+1)/2
  if(d>1)  A[!lower.tri(A,diag=TRUE)] <- param[(p*d+d+1):i0]
  eta <- param[(i0 + 1):(i0 + d)]
  nu <- if(length(param) == (i0 + d + 1)) exp(param[i0 + d + 1]) else NULL
  Oinv <- t(A) %*% diag(D,d,d) %*% A
  # Omega <- pd.solve(Oinv)
  Ainv <- backsolve(A, diag(d))
  Omega <- Ainv %*% diag(1/D,d,d) %*% t(Ainv)
  Omega <- (Omega + t(Omega))/2
  omega <- sqrt(diag(Omega))
  alpha <- eta * omega
  dimnames(beta) <- list(x.names, y.names)
  dimnames(Omega) <- list(y.names, y.names)
  if (length(y.names) > 0) names(alpha) <- y.names
  dp <- list(beta=beta, Omega=Omega, alpha=alpha)
  if(!is.null(nu)) dp$nu <- nu
  list(dp=dp, beta=beta, Omega=Omega, alpha=alpha, nu=nu, Omega.inv=Oinv,
     log.det=sum(log(D)))
}


dplist2optpar <- function(dp,  Omega.inv=NULL)
{# convert DP list to vector form of optimization parameters 
  beta <- dp[[1]]
  Omega <- dp[[2]]
  alpha <- dp[[3]]
  d <- length(alpha)
  nu <- dp$nu
  eta <- alpha/sqrt(diag(Omega))
  Oinv <- if(is.null(Omega.inv)) pd.solve(Omega) else Omega.inv
  if(is.null(Oinv)) stop("matrix Omega not symmetric positive definite")
  upper <- chol(Oinv)
  D <- diag(upper)
  A <- upper/D
  D <- D^2
  param <- if(d > 1)  c(beta, -log(D)/2, A[!lower.tri(A, diag = TRUE)], eta)
     else c(beta, -log(D)/2, eta)
  if(!is.null(dp$nu))  param <- c(param, log(dp$nu)) 
  param <- as.numeric(param)
  #attr(param, 'ind') <- cumsum(c(length(beta), d, d*(d-1)/2, d, length(dp$nu)))
  # param <- c(param, dp$nu)
  return(param) 
}

mst.pdev.grad <- function(param, x, y, w, fixed.nu=NULL, symmetr=FALSE, 
                          penalty=NULL, trace=FALSE)
{ # based on Appendix B of Azzalini & Capitanio (2003, arXiv-0911.2342)
  # except for a few quite patent typos (transposed matrices, etc)
  if(missing(w)) w <- rep(1,nrow(y))
  d <- ncol(y)
  p   <- ncol(x)
  beta<- matrix(param[1:(p*d)],p,d)
  D  <- exp(-2*param[(p*d+1):(p*d+d)])
  A  <- diag(d)
  i0 <- p*d + d*(d+1)/2
  if(d>1) A[!lower.tri(A,diag=TRUE)] <- param[(p*d+d+1):i0]
  eta  <- if(symmetr) rep(0,d) else param[(i0+1):(i0+d)]
  nu   <- if(is.null(fixed.nu)) exp(param[length(param)]) else fixed.nu
  Oinv <- t(A) %*% diag(D,d,d) %*% A
  u    <- y - x %*% beta
  u.w  <- u * w
  Q    <- as.vector(rowSums((u %*% Oinv) * u.w))
  L    <- as.vector(u.w %*% eta)
  sf   <- if(nu < 1e4) sqrt((nu+d)/(nu+Q)) else sqrt((1+d/nu)/(1+Q/nu))
  t.   <- L*sf                                     # t(L,Q,nu) in \S 5.1
  # dlogft<- (-0.5)*(1+d/nu)/(1+Q/nu)              # \tilde{g}_Q
  dlogft <- (-0.5)*sf^2                            # \tilde{g}_Q, again
  dt.dL <- sf                                      # \dot{t}_L
  dt.dQ <- (-0.5)*L*sf/(Q+nu)                      # \dot{t}_Q
  logT. <- pt(t., nu+d, log.p=TRUE)
  dlogT.<- exp(dt(t., nu+d, log=TRUE) - logT.)     # \tilde{T}_1
  Dbeta <- (-2* t(x) %*% (u.w*dlogft) %*% Oinv 
            - outer(as.vector(t(x) %*% (dlogT. * dt.dL* w)), eta)
            - 2* t(x) %*% (dlogT.* dt.dQ * u.w) %*% Oinv )
  Deta  <- colSums(dlogT.*sf*u.w)
  if(d>1) {
     M  <- 2*( diag(D,d,d) %*% A %*% t(u * dlogft
               + u * dlogT. * dt.dQ) %*% u.w)
     DA <- M[!lower.tri(M,diag=TRUE)]
     }
  else DA<- NULL
  M <- (A %*% t(u*dlogft + u*dlogT.*dt.dQ) %*% u.w %*% t(A))
  if(d>1) DD <- diag(M) + 0.5*sum(w)/D
     else DD <- as.vector(M + 0.5*sum(w)/D) 
  grad <- (-2) * c(Dbeta, DD*(-2*D), DA, if(!symmetr) Deta)
  if(is.null(fixed.nu)) {
    df0 <- min(nu, 1e8)
    if(df0 < 10000){
       diff.digamma <- digamma((df0+d)/2) - digamma(df0/2)
       log1Q<- log(1+Q/df0)
     }
    else
      {
       diff.digamma <- log1p(d/df0)
       log1Q <- log1p(Q/df0)
      }
    dlogft.ddf <- 0.5 * (diff.digamma - d/df0
                        + (1+d/df0)*Q/((1+Q/df0)*df0) - log1Q)
    eps   <- 1.0e-4
    df1 <- df0 + eps
    sf1 <- if(df0 < 1e4) sqrt((df1+d)/(Q+df1)) else sqrt((1+d/df1)/(1+Q/df1))
    logT.eps <- pt(L*sf1, df1+d, log.p=TRUE)
    dlogT.ddf <- (logT.eps-logT.)/eps
    Ddf   <- sum((dlogft.ddf + dlogT.ddf)*w)
    grad <- c(grad, -2*Ddf*df0)
    #browser()
    }
  if(!is.null(penalty)) { 
    if(symmetr) stop("penalized log-likelihood not allowed when alpha=0")
    Ainv <- backsolve(A, diag(d))
    Omega <- Ainv %*% diag(1/D,d,d) %*% t(Ainv)
    omega <- diag(Omega)
    alpha <- eta*omega
    Q <- Qpenalty(list(alpha, cov2cor(Omega)), nu, der=1)
    comp <-  1:(length(alpha)+is.null(fixed.nu))
    Qder <- attr(Q, "der1") * c(1/omega, 1)[comp] 
    # gradient for transformed variable (alpha --> eta)
    grad <- grad + 2*c(rep(0, p*d + d*(d+1)/2),  Qder)
    }
  if(trace) cat("mst.pdev.grad: norm is ", format(sqrt(sum(grad^2))), "\n")
  return(grad)
}


n = 5000 # Number of observations
k = 3    # Dimension of MvSkewTDist
p = 4    # Number of covariates

β = matrix(rnorm(p*k), nrow=p, ncol=k)
X = matrix(rnorm(n*p), nrow=n, ncol=p)
Ω = crossprod(matrix(rnorm(k*k), nrow=k, ncol=k))
α = rnorm(k)
ν = 4.0

Y = rmst(n=n, Omega=Ω, alpha=α, nu=ν) + X%*%β

write.table(X,"x.txt", row.names=F, col.names=F)
write.table(Y, "y.txt", row.names=F, col.names=F)

#dp = mst.mple(X,Y)$dp


# Initial parameters for fit
linfit = lm.fit(X,Y,singular.ok=F)
βinit = coef(linfit)
Ωinit = var(resid(linfit))
αinit = rep(1.0,k)

# Write data and initial data to files
write.table(X,"x.txt", row.names=F, col.names=F)
write.table(Y, "y.txt", row.names=F, col.names=F)
write.table(βinit, "Beta.txt", row.names=F, col.names=F)
write.table(Ωinit, "Omega.txt", row.names=F, col.names=F)

dp = list()
dp$beta = βinit
dp$Omega = Ωinit
dp$alpha = αinit
dp$nu = 4.0

param=dplist2optpar(dp)
cat("Parameter vector:", param, "\n")
nll = mst.pdev(param, X, Y)
cat("Neg. Log Likelihood: ", nll, "\n")
grad = mst.pdev.grad(param, X, Y)
cat("Gradient: ", grad, "\n")

mst.mple(X,Y,opt.method="BFGS")
#lm.fit(X, Y, singular.ok=FALSE)$coefficients
