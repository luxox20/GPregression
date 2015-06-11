rm(list=ls())
source("GP.R")
library("numDeriv")
n=100
p=5
X=matrix(rnorm(n*p),ncol=p)
beta=2^(0:(1-p))
alpha=3
tau=2
eps=rnorm(n,0,1/sqrt(tau))
y=alpha+as.vector(X%*%beta + eps)
gp<-gp.init(X,y)

Linn = function(para){
    gp<-gp.setparams(gp,para)
    fit <-gp.loglike(gp)      
    return(-fit)
  }
#print("numerical derivative:")
#print(grad(Linn,gp.getparams(gp)))
#print("exact derivative:")
#print(gp.grad(gp))
#Ñ!
gp<-gp.hmc(gp,niter=1000,leapfrog=3,100)
pred<-gp.pred(gp,X)
rmse<-sum((y-pred$mean)^2)/n
print(rmse)
gp<-gp.optim(gp)
pred<-gp.pred(gp,X)
rmse<-sum((y-pred$mean)^2)/n
print(rmse)
