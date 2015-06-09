rm(list=ls())
setwd("/Users/negativland/Google Drive/NIR wood classification/code")
source("GP.R")
n=500
p=20
X=matrix(rnorm(n*p),ncol=p)
beta=2^(0:(1-p))
alpha=3
tau=2
eps=rnorm(n,0,1/sqrt(tau))
y=alpha+as.vector(X%*%beta + eps)
gp<-gp.init(X,y)
fin<-TRUE
while(fin){
  gp<-gp.sample(gp)
  fin<-!is.finite(gp.loglike(gp))
}
Linn = function(para){
    gp<-gp.setparams(gp,para)
    fit <-gp.loglike(gp)      
    return(-fit)
  }
print("numerical derivative:")
print(grad(Linn,gp.getparams(gp)))
print("exact derivative:")
print(gp.grad(gp))
#gp<-gp.hmc(gp,niter=100,leapfrog=10,50)
