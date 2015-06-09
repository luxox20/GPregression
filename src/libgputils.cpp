#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <random> 

using namespace Eigen;
using namespace std;

extern "C"{
  #include <math.h>
  #include <R.h>

  
  void squared_exponential(double* XData, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* fpar, double* nu, int* dim, double* result){
	Map<MatrixXd>X(XData,*Xnrow,*Xncol);
	Map<MatrixXd>Y(Ydata,*Ynrow, *Yncol);
    MatrixXd Cov;
    Cov.resize(*Xnrow,*Ynrow);
		for(int i=0; i < *Xnrow; ++i){			
			for(int j=0; j < *Ynrow; ++j){
				double d,S=0;
				for(int k=0; k < *dim; ++k){				
					d = X(i,k) - Y(j,k);
					S += nu[k] * pow(d,2.0);
				}				
				Cov(i,j) =  exp(*fpar - 0.5 * S);
			}
		}
    memcpy(result,Cov.data(),Cov.size() * sizeof(double));
	}	
	
  void gp_predict (double* Kxsdata, int* Kxsnrow, int* Kxsncol,double*  Kxxdata, int* Kxxnrow, int* Kxxncol, double*  Ydata, int* Ynrow, double* MeanPred){
    Map<MatrixXd,RowMajor>Kxs(Kxsdata,*Kxsnrow,*Kxsncol);
    Map<MatrixXd>Kxx(Kxxdata,*Kxxnrow,*Kxxncol);
    Map<MatrixXd>Y(Ydata,*Ynrow,1);
    LLT<MatrixXd> Chol(Kxx);
    VectorXd alpha=Chol.solve(Y);
    VectorXd Pred = Kxs.transpose()*alpha;
    memcpy(MeanPred,Pred.data(),Pred.size() * sizeof(double));
  }
  
  void gp_predict_var (double* Kssdata, int* Kssnrow, int* Kssncol, double* Kxsdata, int* Kxsnrow, int* Kxsncol,double*  Kxxdata, int* Kxxnrow, int* Kxxncol, double* VarPred){
    Map<MatrixXd>Kss(Kssdata,*Kssnrow,*Kssncol);
    Map<MatrixXd>Kxs(Kxsdata,*Kxsnrow,*Kxsncol);
    Map<MatrixXd>Kxx(Kxxdata,*Kxxnrow,*Kxxncol);
    LLT<MatrixXd> Chol(Kxx);
    VectorXd alpha=Chol.solve(Kxs);
    MatrixXd Var=Kss- Kxs.transpose() * Kxx.inverse()*Kxs;
    memcpy(VarPred,Var.data(),Var.size() * sizeof(double));
  }
    
  void squared_exponential_geo (double* XData, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* nu, int* dim, double* result){
		Map<MatrixXd>X(XData,*Xnrow,*Xncol);
		Map<MatrixXd>Y(Ydata,*Ynrow, *Yncol);
		int i, j,index=0;
		double r = 6378;
		double S,d;
		for(i=0; i < *Xnrow; i++){
			for(j=0; j < *Ynrow; ++j){
				d = 2*r*asin(sqrt( pow(((sin((X(i,0) - Y(j,0))/2))),2.0) + cos(X(i,0))*cos(Y(j,0))*pow(((sin((X(i,1) - Y(j,1))/2))),2.0)));
				S = *nu * d;
				result[index++] = exp(-0.5 * S);
			}
		}	
	}

void log_likelihood(double* Kxxdata, int* Kxxnrow, int* Kxxncol,double* Ydata, int* Ynrow, double* result){
		Map<MatrixXd>Kxx(Kxxdata,*Kxxnrow,*Kxxncol);
		Map<VectorXd>Y(Ydata,*Ynrow);
        LLT<MatrixXd> Chol(Kxx);
		ArrayXd loglike = -.5*Y.transpose()*Chol.solve(Y);
		loglike += -0.5*log(Kxx.determinant())-(0.5* *Ynrow * log(2*M_PI));			
		result[0] = loglike[0];
	}

  void gp_grad(double* XData,int* XDatanrow,int* XDatancol,double* YData,int* YDatanrow,int* YDatancol,double* KxxData,int* KxxDatanrow,int* KxxDatancol,double* fparData,double* noiseData,double* biasData,double* nuData,double* resultado_grad){
    Map<MatrixXd>Kxx(KxxData,*KxxDatanrow,*KxxDatancol);
    Map<MatrixXd>X(XData,*XDatanrow,*XDatancol);
    Map<VectorXd>Y(YData,*YDatanrow,*YDatancol);
    Map<VectorXd>nu(nuData,*XDatancol);
    MatrixXd unos = MatrixXd::Ones(*XDatanrow,*XDatanrow);
    MatrixXd fac = exp(*biasData)*unos;
    MatrixXd Sigma=Kxx+exp(*noiseData)*MatrixXd::Identity(*XDatanrow,*XDatanrow);
    Sigma+=fac;
    MatrixXd invSigma=Sigma.inverse();
    VectorXd cninvt=invSigma*Y;
    VectorXd grad=VectorXd::Zero(3+*XDatancol);
    // v0 gradiente
   double gfpar = (invSigma*Kxx).trace() - (cninvt.transpose())*Kxx*cninvt;      


    double gbias = (invSigma*fac).trace() - (cninvt.transpose())*fac*cninvt;
    
    // inputs gradient
    double gnoise = exp(*noiseData)*(invSigma.trace() - (cninvt.transpose())*cninvt);  

    //inputs gradient
    MatrixXd matx;  
    MatrixXd dmat;
    VectorXd gnu(*XDatancol); 
    MatrixXd aux1 = MatrixXd::Ones(1,*XDatanrow);
    MatrixXd aux2 = MatrixXd::Ones(*XDatanrow,1);
    
    for(int k=1;k<(*XDatancol)+1;k++){
      matx.noalias() = (X.block(0,0,*XDatanrow,k).cwiseProduct(X.block(0,0,*XDatanrow,k)))*aux1 - 2*(X.block(0,0,*XDatanrow,k))*((X.block(0,0,*XDatanrow,k)).transpose()) + aux2*((X.block(0,0,*XDatanrow,k)).cwiseProduct(X.block(0,0,*XDatanrow,k)).transpose());
      dmat.noalias() = -0.5*exp(nu(k-1))*(Kxx.cwiseProduct(matx));
      gnu(k-1) = 0.5*((invSigma*dmat).diagonal().sum()) - (cninvt.transpose())*dmat*cninvt;
    }
    grad << gfpar,gnoise,gbias,gnu;
    grad = 0.5*grad;
    memcpy(resultado_grad,grad.data(),grad.size()*sizeof(double));
  }
  
  void gp_hmc(int* niter,int* leapfrogData,int* burninData,double* init_parData,int* length_init_par,double* KxxData,int* Kxxnrow,int* Kxxncol,double* XData,int* XDatanrow,int* XDatancol,double* YData,int* YDatanrow,int* YDatancol,double* invSigmaData,int* invSigmanrow,int* invSigmancol,double* trcinvData,double* cninvtData,int* cninvtnrow,double* noiseData,double* biasData,double* nu,double* resultado_hmc,int* resultado_rate){
    Map<VectorXd>init_par(init_parData,*length_init_par);
    MatrixXd samples = MatrixXd::Zero(*niter+1,*length_init_par);
    int data_dim=(*length_init_par)-3;
    VectorXd p0(*length_init_par);
    VectorXd x0(*length_init_par);       
    VectorXd pstar(*length_init_par);
    VectorXd pstar_n(*length_init_par);
    VectorXd xstar(*length_init_par);    
    VectorXd leap_grad(*length_init_par);
    double aleatorio, delta, resultado_grad[*length_init_par], 
        resultado_likelihood_1, resultado_likelihood_2;
    double U0, UStar, K0, KStar, alpha, uniforme;
    int rate=0, lambda, i, j;
    samples.row(1) = init_par;
    default_random_engine generator;
    uniform_real_distribution<double> runif(0.0,1.0);
    normal_distribution<double> rnormal(0.0,1.0);
    bernoulli_distribution rbernoulli(0.25);
    aleatorio = rnormal(generator);
            
    if(aleatorio < 0.5)
      lambda = -1;    
    else
      lambda = 1;
    delta = lambda*0.02*(1.0+0.1*runif(generator));
    
    for(i=2;i<(*niter);i++){     

      for(int p=0;p<(*length_init_par);p++)
        p0(p) = rnormal(generator);

      x0 = init_par;
      //memcpy ( nu, x0.tail(data_dim).data(), data_dim* sizeof(double) );    
      //gp_grad(KxxData,Kxxnrow,Kxxncol,XData,XDatanrow,XDatancol,invSigmaData,invSigmanrow,invSigmancol,trcinvData,cninvtData,cninvtnrow,&x0(1),&x0(2),x0.tail(data_dim).data(),&resultado_grad[0]);
      Map<VectorXd>grad(&resultado_grad[0],*length_init_par);
    
      pstar.head(3) = p0.head(3) - (delta/2)*grad.head(3);
      xstar.head(3) = x0.head(3) + delta*pstar.head(3);
      xstar.tail(data_dim)=x0.tail(data_dim);
      for(int d=3;d<data_dim;d++){
        xstar[d]=rbernoulli(generator);
      }  
      for(j=1;j<(*leapfrogData)-1;j++){
          try{
            squared_exponential(XData,XDatanrow,XDatancol,XData,XDatanrow,XDatancol,&xstar(0),xstar.tail(data_dim).data(),&data_dim,KxxData);    
            //gp_grad(KxxData,Kxxnrow,Kxxncol,XData,XDatanrow,XDatancol,invSigmaData,invSigmanrow,invSigmancol,trcinvData,cninvtData,cninvtnrow,&xstar(1),&xstar(2),xstar.tail(data_dim).data(),&resultado_grad[0]);         
            Map<VectorXd>grad2(&resultado_grad[0],*length_init_par);
            leap_grad = grad2;
          }
          catch(exception e){
            squared_exponential(XData,XDatanrow,XDatancol,XData,XDatanrow,XDatancol,&x0(0),x0.tail(data_dim).data(),&data_dim,KxxData);       
            //gp_grad(KxxData,Kxxnrow,Kxxncol,XData,XDatanrow,XDatancol,invSigmaData,invSigmanrow,invSigmancol,trcinvData,cninvtData,cninvtnrow,&x0(1),&x0(2),x0.tail(data_dim).data(),&resultado_grad[0]);       
            Map<VectorXd>grad2(&resultado_grad[0],*length_init_par);
            leap_grad = grad2;
          }              
          pstar.head(3) = pstar.head(3) - delta*leap_grad.head(3) ;
          xstar.head(3)  = xstar.head(3)  + delta*pstar.head(3) ;
      }
      //for(int d=4;d<(*length_init_par-3);d++){
      //      xstar[d]=rbernoulli(generator);
      //}  
      cout << "x_star:" << xstar.transpose() << endl;
          
      //gp_grad(KxxData,Kxxnrow,Kxxncol,XData,XDatanrow,XDatancol,invSigmaData,invSigmanrow,invSigmancol,trcinvData,cninvtData,cninvtnrow,&xstar(1),&xstar(2),xstar.tail(data_dim).data(),&resultado_grad[0]);  
      Map<VectorXd>grad3(&resultado_grad[0],*length_init_par);      
      pstar = pstar - (delta/2)*grad3;
           
      log_likelihood(KxxData,Kxxnrow,Kxxncol,YData,YDatanrow,&resultado_likelihood_1);
      U0 = (-1)*resultado_likelihood_1;
         
      log_likelihood(KxxData,Kxxnrow,Kxxncol,YData,YDatanrow,&resultado_likelihood_2);
      UStar = (-1)*resultado_likelihood_2;

      K0 = p0.transpose()*(p0*0.5);
      pstar_n = (-1)*pstar;
      KStar = pstar_n.transpose()*(pstar_n*0.5);
      alpha = min(1.0,exp((U0+K0)-(UStar+KStar)));
       
      uniforme = runif(generator);
      if(uniforme<alpha){
          samples.block(i,0,1,*length_init_par) = xstar.transpose();
          rate = rate+1;
          x0=xstar;
      }
      else{
          samples.block(i,0,1,*length_init_par) = samples.block(i-1,0,1,*length_init_par);
      }
    }
    
    memcpy(resultado_hmc,samples.data(),samples.size()*sizeof(double));
    resultado_rate[0] = rate;  
  }
}
