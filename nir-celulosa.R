rm(list=ls())
library(pls)
library(ggplot2)
library(MASS)
source("pca.R")
source("GP.R")

concatenate<-function(file_name,data){
    tmp<-read.table(file=file_name,header=T)
    ind<-paste(tmp[,3])!="-"
    row_data<-as.numeric(paste(tmp[ind,3]))
    data<-rbind(data,row_data)
    colnames(data)<-tmp[ind,1]
    return(data)
}

concatenate.list<-function(file_name,lista){
  tmp<-read.table(file=file_name,header=T)
  lista<- c(lista,colnames(tmp)[3])
  return(lista)
}
NIR<-data.frame()
referencias<-read.csv(file="../data/DAT/referencias.csv",header=T)

files<-paste0("../data/DAT/",paste0(unlist(strsplit(as.character(referencias$file_name),split=".spc")),".dat"))
for (i in 1:length(files)) {
  NIR<-concatenate(files[i],NIR)
}

wood<-data.frame(NIR=matrix(),referencias[2],referencias[3])
wood$NIR<-as.matrix(NIR)
rm(files,i,concatenate,concatenate.list,NIR,referencias)

# PLS, KPLS
idx<-c(1:40)
idxtest<-c(41:58)
X<-wood$NIR[idx,]
y<-wood$lig_ref[idx]
Xtest<-wood$NIR[idxtest,]
ytest<-wood$lig_ref[idxtest]

test.method<-function(Z=1){
  wood.pls <- mvr(y~X,ncomp=Z,method = "simpls", scale=TRUE,validation = "LOO")
  wood.kpls <- plsr(y~X,ncomp=Z,method = "kernelpls",scale=TRUE, validation = "LOO")
  r2.pls<-R2(wood.pls,estimate="all")
  r2.kpls<-R2(wood.kpls,estimate="all")
  y.pls<-predict(wood.pls,newdata=Xtest,comp=Z)
  y.kpls<-predict(wood.kpls,newdata=Xtest,comp=Z)
  # Bayesian non parametric - Gaussian Process regression
  pred<-gp.pca(X,y,Z,Xtest)
  
  rms.gp<-list(train=pca.rmse(y,pred$y_train),test=pca.rmse(ytest,pred$y_test))
  rms.pls<-list(train=pca.rmse(y,wood.pls$fitted.values[,,Z]),test=pca.rmse(ytest,y.pls))
  rms.kpls<-list(train=pca.rmse(y,wood.kpls$fitted.values[,,Z]),test=pca.rmse(ytest,y.kpls))
  
  r2.gp<-list(train=pca.r2(y,pred$y_train),test=pca.r2.test(y,ytest,pred$y_test))
  r2.pls<-list(train=pca.r2(y,wood.pls$fitted.values[,,Z]),test=pca.r2.test(y,ytest,y.pls))
  r2.kpls<-list(train=pca.r2(y,wood.kpls$fitted.values[,,Z]),test=pca.r2.test(y,ytest,y.kpls))
  
  
  print(cbind(rms.gp,rms.kpls,rms.pls))
  
  print(cbind(r2.gp,r2.kpls,r2.pls))
  
  ntest<-length(ytest)
  test.dat <- data.frame(set=rep("test",ntest*3),method = rep(c("GP", "KPLS","PLS"), each=ntest),reference=rep(ytest,3),prediction=rbind(pred$y_test,matrix(y.kpls),matrix(y.pls)))
  pp<-ggplot(test.dat, aes(x=reference, y=prediction)) + geom_point(shape=1)
  pp<-pp+geom_smooth(method=lm)+facet_grid(. ~ method)
  ggsave(file=paste0("figures/lignina_test_",Z,".pdf"),plot=pp,scale=2)
  
  ntrain<-length(y)
  train.dat <- data.frame(set=rep("train",ntrain*3),method = rep(c("GP", "KPLS","PLS"), each=ntrain),reference=rep(y,3),prediction=rbind(pred$y_train,matrix(wood.kpls$fitted.values[,,Z]),matrix(wood.pls$fitted.values[,,Z])))
  pp<-ggplot(train.dat, aes(x=reference, y=prediction)) + geom_point(shape=1)
  pp<-pp+geom_smooth(method=lm)+facet_grid(. ~ method)
  ggsave(file=paste0("figures/lignina_train_",Z,".pdf"),plot=pp,scale=2)
}

for (i in (1:3)) test.method(i)