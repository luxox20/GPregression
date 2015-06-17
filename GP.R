dyn.load("src/libgputils.so")



gp.init<-function(xtrain,ytrain,option=""){
  gp<-list()
  gp<-gp.setdata(gp,xtrain,ytrain)
  d<-dim(gp$xtrain)[2]
  #parameters of covariance function (squared exponential)
  gp$bias<-0
  gp$sigma_y<-0
  gp$nu<-rep(0,d)
  gp$min_noise <- sqrt(.Machine$double.eps); #minimun allowed output
  gp$sigma_f<-0 #alpha
  gp$nwts<-3 + d # number of parameters of covariance function  
  gp$opt=option
  fin<-TRUE
  while(fin){
    gp<-gp.sample(gp)
    fin<-!is.finite(gp.loglike(gp))
  }
  return(gp)
}

gp.plsa<-function(gp,Z,maxiter,tol){
  n<-dim(gp$xtrain)[1]
  d<-dim(gp$xtrain)[2]
  pxy<-matrix(0,nrow=n,ncol=d)
  pxgz<-matrix(0,nrow=n,ncol=Z)
  pygz<-matrix(0,nrow=d,ncol=Z)
  pz<-rep(0,Z)
  Fun<-.C("plsa",as.double(gp$xtrain),as.integer(n),as.integer(d),as.integer(Z),as.integer(maxiter),as.double(tol),pxy=as.double(pxy),pygz=as.double(pygz),pxgz=as.double(pxgz),pz=as.double(pz))    
  pxy<-matrix(Fun$pxy,,nrow=n,ncol=d)
  pxgz<-matrix(Fun$pxgz,nrow=n,ncol=Z)
  pygz<-matrix(Fun$pygz,nrow=d,ncol=Z) 
  return(list(pxy=pxy,pxgz=pxgz,pygz=pygz,pz=Fun$pz))
}

gp.setdata<-function(gp,xtrain,ytrain){
  yy<-gp.scale(ytrain)
  ytrain<-yy$dato
  gp$my<-yy$my
  gp$sy<-yy$sy
  gp$xtrain<-as.matrix(xtrain)
  gp$ytrain<-as.matrix(ytrain)
  return(gp)
}

gp.sample<-function(gp){
  d<-dim(gp$xtrain)[2]
  mu<-1*d^(2)
  gp$sigma_f<-rnorm(1,0,1)
  gp$sigma_y<-rnorm(1,0,1)
  gp$nu<-rnorm(d,0,1)
  gp$bias<-rnorm(1,0,1)
  return(gp)
}

gp.covar<-function(gp,x1,x2){
  if(!is.matrix(x1)) x1<-as.matrix(x1) 
  if(!is.matrix(x2)) x2<-as.matrix(x2)  
  n <- dim(x1)[1]
  d <- dim(x1)[2]
  m<-dim(x2)[1]
  nu<-gp$nu
  sigma_f<-gp$sigma_f
  K<-matrix(0,nrow=n,ncol=m)
  Fun<-.C("squared_exponential",as.double(x1),as.integer(n),as.integer(d),as.double(x2),as.integer(m),as.integer(d),as.double(sigma_f),as.double(nu),as.integer(d),K=as.double(K))    
  K<-matrix(Fun$K,nrow=n,ncol=m)
  return (K)
}

gp.pred<-function(gp,xtest){
    xtest<-as.matrix(xtest)
    n<-dim(gp$xtrain)[1]
    d<-dim(gp$xtrain)[2]
    m<-dim(xtest)[1]
    Kss<-gp.covar(gp, xtest, xtest)+ exp(gp$bias)
    Kxx<-gp.covar(gp,gp$xtrain,gp$xtrain)+ exp(gp$bias)
    Sigma<- Kxx + (gp$min_noise + exp(gp$sigma_y))*diag(n) 
    Kxs<- gp.covar(gp,gp$xtrain,xtest)+ exp(gp$bias)
    pred.mean<-matrix(0,m,1)
    pred.var<-matrix(0,m,m) 
    nkxs<-dim(Kxs)
    nkxx<-dim(Sigma)
    nkss<-dim(Kss)
    nyt<-dim(gp$ytrain)
    Fun<-.C("gp_predict",as.double(Kxs),as.integer(nkxs[1]),as.integer(nkxs[2]),
       as.double(Sigma),as.integer(nkxx[1]),as.integer(nkxx[2]),
       as.double(gp$ytrain),as.integer(nyt[1]),MeanPred=as.double(pred.mean))
    pred.mean<-matrix(Fun$MeanPred,nrow=m,ncol=1)
    Fun2<-.C("gp_predict_var",as.double(Kss),as.integer(nkss[1]),as.integer(nkss[2]),
       as.double(Kxs),as.integer(nkxs[1]),as.integer(nkxs[2]),as.double(Sigma),
       as.integer(nkxx[1]),as.integer(nkxx[2]),VarPred=as.double(pred.var))
    pred.var<-matrix(Fun2$VarPred,nrow=m,ncol=m,byrow=T)
    return(list(mean=pred.mean*gp$sy+gp$my,variance=pred.var,Kxs=Kxs))
}

gp.loglike<-function(gp){
	n<-dim(gp$xtrain)[1]
	Kxx<-gp.covar(gp,gp$xtrain,gp$xtrain)
	ll<-0
	nkxx<-dim(Kxx)
  nyt<-dim(gp$ytrain)	
	Fun<-.C("log_likelihood",as.double(Kxx),
        as.integer(nkxx[1]),as.integer(nkxx[2]),
        as.double(gp$ytrain),as.integer(nyt[1]),as.double(gp$sigma_f),as.double(gp$sigma_y),as.double(gp$bias),ll=as.double(ll))
    ll=Fun$ll#+gp.prior(gp)
	return(as.numeric(ll))	
}

gp.prior<-function(gp){
  par<-gp.getparams(gp)
  return(sum(dnorm(par,0,1,log=T)))
}

gp.optim <-function(gp){
  init.par<-gp.getparams(gp)
  Linn = function(para){
    gp<-gp.setparams(gp,para)
    fit <-gp.loglike(gp)      
    return(-fit)
  } 
  Grad = function(para){
    gp<-gp.setparams(gp,para)
    #grad<-gp.grad(gp)
    grad<-grad(Linn,gp.getparams(gp))
    return(grad)
  }
  (est = optim(init.par, Linn, gr=Grad, method="BFGS", hessian=FALSE))
  gp$sigma_f<-est$par[1]
  gp$sigma_y<-est$par[2]
  gp$bias<-est$par[3]
  gp$nu<-est$par[4:length(est$par)]
  return(gp)
}

#gp.hmc <-function(gp,niter=1000,leapfrog=10,burnin=500)
gp.hmc<-function(gp,niter,leapfrog,burnin){
    Kxx<-gp.covar(gp,gp$xtrain,gp$xtrain) #covf
    init.par<-gp.getparams(gp)
    samples <- matrix(0,nrow=niter,ncol=length(init.par));
    rate <- c(0);    
    Fun <- .C("gp_hmc",as.integer(niter),as.integer(leapfrog),as.integer(burnin),as.double(Kxx),as.integer(dim(Kxx)[1]),as.integer(dim(Kxx)[2]),as.double(gp$xtrain),as.integer(dim(gp$xtrain)[1]),as.integer(dim(gp$xtrain)[2]),as.double(gp$ytrain),as.integer(dim(gp$ytrain)[1]),as.integer(dim(gp$ytrain)[2]),as.double(init.par),as.integer(length(init.par)),samples=as.double(samples),rate=as.integer(rate));
    rate <- as.integer(Fun$rate);  
    samples <- matrix(Fun$samples,nrow=niter,ncol=length(init.par));
    print(samples)
    #samples <- matrix(Fun$samples,nrow=5,ncol=length(init.par));
    cat('Acceptance Rate : ',    100*rate/niter,'%\n')
    post.par<-rep(0,length(init.par))
    post.par<-colMeans(samples[-1:-burnin,])
    #post.par[-1:-3]<-rep(0,d)
    gp<-gp.setparams(gp,post.par)
    return(gp)
}

gp.grad<-function(gp){
	n<-dim(gp$xtrain)[1]
	d<-dim(gp$xtrain)[2]
    Kxx<-gp.covar(gp,gp$xtrain,gp$xtrain);
    grad <- rep(0,3+d)
    Fun <- .C("gp_grad",as.double(gp$xtrain),as.integer(n),as.integer(d),as.double(gp$ytrain),as.integer(dim(gp$ytrain)[1]),as.integer(dim(gp$ytrain)[2]),as.double(Kxx),as.integer(dim(Kxx)[1]),as.integer(dim(Kxx)[2]),as.double(gp$f_par),as.double(gp$sigma_y),as.double(gp$bias),as.double(gp$nu),grad=as.double(grad));
    grad <- as.vector(Fun$grad)
    return(grad)
}

gp.getparams<-function(gp){
    c(gp$sigma_f,gp$sigma_y,gp$bias,gp$nu)
}

gp.setparams<-function(gp,para){
    #print(para)
    gp$sigma_f<-para[1]
    gp$sigma_y<-para[2]
    gp$bias<-para[3]
    gp$nu<-para[4:length(para)]
    return(gp)
}

gp.scale<-function(dato){
  #dato<-(dato-min(dato))/(max(dato)-min(dato));
  my<-mean(dato)
  sy<-sd(dato)
  dato<-(dato-my)/(sy)
  return(list(dato=dato,my=my,sy=sy))
}

trace<-function(m){
  return (sum(diag(m)))
}
