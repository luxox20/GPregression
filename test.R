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

source("pca.R")
x1<-rnorm(n,0,3)
x2<-2*x1*rnorm(n,0,1)
x3<-rnorm(n,10,1)
y<-x1+2*x2+.1*x3+rnorm(n,0,.5)
X<-cbind(x1,x2,x3)

fit<-lm(y~X)
rmse1<-sum((y-predict(fit,data.frame(X)))^2)/length(y)
print("RMSE1:")
print(rmse1)
y_pred<-pcr(X,y,1)
rmse2<-sum((y-y_pred)^2)/length(y)
print("RMSE2:")
print(rmse2)
fit<-lm(y~X[,1:3])
rmse3<-sum((y-predict(fit,data.frame(X[,1])))^2)/length(y)
print("RMSE3:")
print(rmse3)