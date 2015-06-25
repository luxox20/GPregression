rm(list=ls())
library(pls)
source("pca.R")
source("GP.R")
data("gasoline")

# PLS, KPLS
Z<-5
idx<-1:50
X<-gasoline$NIR[idx,]
y<-gasoline$octane[idx]
Xtest<-gasoline$NIR[-idx,]
ytest<-gasoline$octane[-idx]

gas.pls <- mvr(y~X,ncomp=Z,method = "simpls", validation = "CV")
gas.kpls <- plsr(y~X,ncomp=Z,method = "kernelpls", validation = "CV")
r2.pls<-R2(gas.pls,estimate="all")
r2.kpls<-R2(gas.kpls,estimate="all")
rms.pls<-RMSEP(gas.pls,comp=Z,)
rms.kpls<-RMSEP(gas.kpls,comp=Z)


pred<-gp.pca(X,y,Z,Xtest)
rms.gp<-list(train=sum((y-pred$y_train)^2)/length(y),test=sum((ytest-pred$y_test)^2)/length(y))


