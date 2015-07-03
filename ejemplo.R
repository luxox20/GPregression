library(GPfit)
library(numDeriv)
source("GP.R")
n = 20; d = 1;
computer_simulator <- function(x) {
y <- log(x+0.1)+sin(5*pi*x)+rnorm(n,0,.5);
return(y)
}
set.seed(1)
library(lhs);
x = maximinLHS(n,d)
y = computer_simulator(x)
xvec <- seq(from=0,to=1,length.out=100);
GPmodel = GP_fit(x,y)
GPprediction = predict.GP(GPmodel,xvec);
yhat = GPprediction$Y_hat;
mse = GPprediction$MSE;
completedata = GPprediction$complete_data;

Linn = function(para){
    gp<-gp.setparams(gp,para)
    fit <-gp.loglike(gp)      
    return(-fit)
  }

gp<-gp.init(x,y)
gp<-gp.hmc(gp,20000,10,10000)
pred<-gp.pred(gp,xvec)
plot(GPmodel)
lines(xvec,pred$mean,cex=.5,pty=10,col="gray60",lwd=1)
