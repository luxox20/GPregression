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
  return(list(rotations=t(y),loadings=pca,center=m,values=v$values[sorted$ix[1:K]]))
}

pcr<-function(X,y,K){
  N<-dim(X)[1]
  D<-dim(X)[2]
  p<-pca(X,K)
  test<-matrix(0,K,N)
  fit<-lm(y~p$rotations)
  for(i in 1:N) test[,i]<-t(p$loadings)%*%(X[i,]-p$center)
  return(predict(fit,as.data.frame(t(test))$fitted.values))
}

gpr<-function(X,y,K,Xtest){
  source("GP.R")
  N<-dim(X)[1]
  T<-dim(Xtest)[1]
  D<-dim(X)[2]
  comp<-pca(X,K)
  fit<-gausspr(comp$rotations, y, kernel="rbfdot")
  #fit<-gp.init(comp$rotations,y)
  #fit<-gp.hmc(fit,20000,3,5000)
  test<-matrix(0,T,K)
  for(i in 1:T) test[i,]<-t(X[1,]-comp$center)%*%comp$loadings
  #pred.train<-gp.pred(fit,comp$rotations)
  #pred.test<-gp.pred(fit,test)
  pred.train<-predict(fit,as.data.frame(comp$rotations))
  pred.test<-predict(fit,as.data.frame(test))
  return(list(y_train=pred.train,y_test=pred.test))
}