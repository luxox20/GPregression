rm(list=ls())
library("numDeriv")
source("GP.R")
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
print("numerical derivative:")
print(grad(Linn,gp.getparams(gp)))
print("exact derivative:")
print(gp.grad(gp))
gp.mc<-gp.hmc(gp,niter=1000,leapfrog=3,500)
pred.mc<-gp.pred(gp.mc,X)
rmse<-sum((y-pred.mc$mean)^2)/n
cat("HMC : ",rmse,"\n")
gp.nr<-gp.optim(gp)
pred.nr<-gp.pred(gp.nr,X)
rmse<-sum((y-pred.nr$mean)^2)/n
cat("Evidence : ",rmse,"\n")
plot(y,pred.mc$mean)
points(y,pred.nr$mean,pch=5,col="red")
