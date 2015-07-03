rm(list=ls())
library(pls)
library(ggplot2)
source("pca.R")
source("GP.R")
data("gasoline")

# PLS, KPLS
Z<-12
idx<-1:50
X<-gasoline$NIR[idx,]
y<-gasoline$octane[idx]
Xtest<-gasoline$NIR[-idx,]
ytest<-gasoline$octane[-idx]

gas.pls <- mvr(y~X,Z,method = "simpls")
gas.kpls <- plsr(y~X,Z,method = "kernelpls")
y.pls<-predict(gas.pls,newdata=Xtest,comp=Z)
y.kpls<-predict(gas.kpls,newdata=Xtest,comp=Z)

pred<-gp.pca(X,y,Z,Xtest)

rms.gp<-list(train=pca.rmse(y,pred$y_train),test=pca.rmse(ytest,pred$y_test))
rms.pls<-list(train=pca.rmse(y,gas.pls$fitted.values[,,Z]),test=pca.rmse(ytest,y.pls))
rms.kpls<-list(train=pca.rmse(y,gas.kpls$fitted.values[,,Z]),test=pca.rmse(ytest,y.kpls))

r2.gp<-list(train=pca.r2(y,pred$y_train),test=pca.r2.test(y,ytest,pred$y_test))
r2.pls<-list(train=pca.r2(y,gas.pls$fitted.values[,,Z]),test=pca.r2.test(y,ytest,y.pls))
r2.kpls<-list(train=pca.r2(y,gas.kpls$fitted.values[,,Z]),test=pca.r2.test(y,ytest,y.kpls))


print(cbind(rms.gp,rms.kpls,rms.pls))

print(cbind(r2.gp,r2.kpls,r2.pls))

ntest<-length(ytest)
test.dat <- data.frame(set=rep("test",ntest*3),method = rep(c("GP", "KPLS","PLS"), each=ntest),reference=rep(ytest,3),prediction=rbind(pred$y_test,matrix(y.kpls),matrix(y.pls)))
#pp<-ggplot(test.dat, aes(x=reference, y=prediction)) + geom_point(shape=1)
#pp<-pp+geom_smooth(method=lm)+facet_grid(. ~ method)
#ggsave(file="gasoline_test.pdf",plot=pp,scale=2)

ntrain<-length(y)
train.dat <- data.frame(set=rep("train",ntrain*3),method = rep(c("GP", "KPLS","PLS"), each=ntrain),reference=rep(y,3),prediction=rbind(pred$y_train,matrix(gas.kpls$fitted.values[,,Z]),matrix(gas.pls$fitted.values[,,Z])))
#pp<-ggplot(train.dat, aes(x=reference, y=prediction)) + geom_point(shape=1)
#pp<-pp+geom_smooth(method=lm)+facet_grid(. ~ method)
#ggsave(file="gasoline_train.pdf",plot=pp,scale=2)

plot(train.dat$reference,train.dat$prediction,pch=21,col=train.dat$method)
