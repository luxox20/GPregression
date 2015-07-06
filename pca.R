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

gp.pca<-function(X,y,K,Xtest){
  source("GP.R")
  N<-dim(X)[1]
  T<-dim(Xtest)[1]
  D<-dim(X)[2]
  comp<-pca(X,K)
  fit<-gp.init(comp$rotations,y)
  #fit<-gp.optim(fit)
  fit<-gp.hmc(fit,5000,0,100)
  test<-matrix(0,T,K)
  for(i in 1:T) test[i,]<-t(Xtest[i,]-comp$center)%*%comp$loadings
  pred.train<-gp.pred(fit,comp$rotations)$mean
  pred.test<-gp.pred(fit,test)$mean
  return(list(y_train=pred.train,y_test=pred.test))
}

gp.plsa<-function(X,y,K,Xtest){
  source("GP.R")
  require("MASS")
  n<-dim(X)[1]
  ntest<-dim(Xtest)[1]
  d<-dim(X)[2]
  plsa<-plsa(rbind(X,Xtest),K,1e2,1e-6)
  fit<-gp.init(plsa$pxgz[1:n,],y)
  fit<-gp.hmc(fit,10000,10,5000)
  pred.train<-gp.pred(fit,plsa$pxgz[1:n,])$mean
  pred.test<-gp.pred(fit,plsa$pxgz[-(1:n),])$mean
  return(list(y_train=pred.train,y_test=pred.test))
}


pca.r2<-function(y,y.pred){
  tss<-sum((y-mean(y))^2)
  rss=sum((y-y.pred)^2)
  return(1-rss/tss)
}

pca.r2.test<-function(y,y_test,y.pred){
  tss<-sum((y-mean(y))^2)
  rss=sum((y_test-y.pred)^2)
  return(1-rss/tss)
}

pca.rmse<-function(y,y.pred){
  rss=sum((y-y.pred)^2)
  return(sqrt(rss/length(y)))
}

pca.rpd<-function(y,y.pred){
  r2<-pca.r2(y,y.pred)
  return((1-r2)^-2)
}

rdirichlet <- function(a) {
  y <- rgamma(length(a), a, 1)
  return(y / sum(y))
}

gen.plsa<-function(x,y,z){
  require(MASS)
  pz<-runif(z)
  pz<-pz/sum(pz)
  pxgz<-matrix(runif(x*z),x,z)
  pxgz %*% diag(1/colSums(pxgz))
  pygz<-matrix(runif(y*z),y,z)
  pygz %*% diag(1/colSums(pygz))
  pxy<-matrix(0,x,y)
  for(i in 1:z) pxy<-pxy+pxgz[,z]%*%t(pygz[,z])*pz[z]
  return(pxy)
}
