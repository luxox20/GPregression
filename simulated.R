library(pls)
x1<-rnorm(100,0,10)
x2<-2*x1
x2<-rnorm(100,1,2)
x3<-rnorm(100,1,2)
x4<-10*x3
x5<-rnorm(100,5,3)
b<-runif(6)
X<-cbind(x1,x2,x3,x4,x5)
y<-cbind(rep(1,100),X)%*%b+rnorm(100,0,1)
pca<-prcomp(X)
fit.pca<-lm(y~pca$x[,1:3])
y_pred<-predict(fit.pca,as.data.frame(pca$x[,1:3]))
cbind(y,y_pred)


