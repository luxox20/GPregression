pca<-function(data,K){
  data<-as.matrix(data)
  N<-dim(data)[1]
  D<-dim(data)[2]
  y<-matrix(0,K,N)
  m<-colMeans(data)
  s<-cov(data)
  v<-eigen(s)
  sorted<-sort(v$values,index.return=TRUE,decreasing=TRUE)
  pca<-v$vectors[,sorted$ix[1:K]]
  for(i in 1:N) y[,i]<-t(pca)%*%(data[i,]-m)
  return(list(rotations=y,loadings=pca,center=m,values=v$values[sorted$ix[1:K]]))
}

pcr<-function(X,y,K,subset){
  N<-dim(X)[1]
  D<-dim(X)[2]
  p<-pca(X[subset,],K)
  test<-matrix(0,K,N)
  fit<-lm(y~t(p$rotations))
  for(i in 1:N) test[,i]<-t(p$loadings)%*%(X[i,]-p$center)
  return(predict(fit,as.data.frame(t(test))$fitted.values))
}

gpr<-function(X,y,K){
  require("GP.R")
  N<-dim(X)[1]
  D<-dim(X)[2]
  p<-pca(X,K)
  fit<-gp.init(t(p$rotations),y)
  fit<-gp.hmc(fit,1000,10,10)
  test<-matrix(0,K,N)
}