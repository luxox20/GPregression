#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <random> 
#include <limits> 

using namespace Eigen;
using namespace std;

extern "C"{
  #include <math.h>
  #include <R.h>

  
  void squared_exponential(double* XData, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* sigma_f, double* nu, int* dim, double* result){
	  Map<MatrixXd>X(XData,*Xnrow,*Xncol);
	  Map<MatrixXd>Y(Ydata,*Ynrow, *Yncol);
    MatrixXd Cov;
    Cov.resize(*Xnrow,*Ynrow);
		for(int i=0; i < *Xnrow; ++i){			
			for(int j=0; j < *Ynrow; ++j){
				double d,S=0;
				for(int k=0; k < *dim; ++k){				
					d = X(i,k) - Y(j,k);
					S += exp(nu[k]) * pow(d,2.0);
				}				
				Cov(i,j) =  exp(*sigma_f - 0.5 * S);
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

void log_likelihood(double* Kxxdata, int* Kxxnrow, int* Kxxncol,double* Ydata, int* Ynrow,double* sigma_f,double* sigma_y,double* bias, double* result){
		Map<MatrixXd>Kxx(Kxxdata,*Kxxnrow,*Kxxncol);
		Map<VectorXd>Y(Ydata,*Ynrow);
    MatrixXd unos = MatrixXd::Ones(*Kxxnrow,*Kxxncol);
    MatrixXd bias_mat = exp(*bias)*unos;
    MatrixXd Sigma=Kxx+(numeric_limits<double>::min()+exp(*sigma_y))*MatrixXd::Identity(*Kxxnrow,*Kxxncol);
    Sigma+=bias_mat;
    LLT<MatrixXd> Chol(Sigma);
		ArrayXd loglike = -.5*Y.transpose()*Chol.solve(Y);
		loglike += -0.5*log(Sigma.determinant())-(0.5* *Ynrow * log(2*M_PI));			
		result[0] = loglike[0];
	}

  void gp_grad(double* XData,int* Xnrow,int* Xncol,double* YData,int* Ynrow,int* Yncol,double* KxxData,int* Kxxnrow,int* Kxxncol,double* sigma_f,double* sigma_y,double* bias,double* nu,double* resultado_grad){
    Map<MatrixXd>Kxx(KxxData,*Kxxnrow,*Kxxncol);
    Map<MatrixXd>X(XData,*Xnrow,*Xncol);
    Map<VectorXd>Y(YData,*Ynrow,*Yncol);
    Map<VectorXd>nu_vec(nu,*Xncol);
    MatrixXd unos = MatrixXd::Ones(*Kxxnrow,*Kxxncol);
    MatrixXd bias_mat = exp(*bias)*unos;
    MatrixXd Sigma=Kxx+(numeric_limits<double>::min()+exp(*sigma_y))*MatrixXd::Identity(*Kxxnrow,*Kxxncol);
    Sigma+=bias_mat;
    MatrixXd invSigma=Sigma.inverse();
    LLT<MatrixXd> Chol(Sigma);
    VectorXd cninvt=Chol.solve(Y);
    VectorXd grad=VectorXd::Zero(3+*Xncol);
    // v0 gradiente
    double g_sigma_f = (invSigma*Kxx).trace() - (cninvt.transpose())*Kxx*cninvt;      
    double g_bias = (invSigma*bias_mat).trace() - (cninvt.transpose())*bias_mat*cninvt;
    // inputs gradient
    double g_sigma_y = (exp(*sigma_y))*(invSigma.trace() - cninvt.dot(cninvt));  
    //inputs gradient
    VectorXd g_nu= VectorXd::Zero(*Xncol); 
    
    for(int l=0;l<*Xncol;l++){
      MatrixXd matx=MatrixXd::Zero(*Xnrow,*Xnrow);  
      MatrixXd dmat=MatrixXd::Zero(*Xnrow,*Xnrow);  ;
      for(int i=0; i < *Xnrow; ++i){      
        for(int j=0; j < *Xnrow; ++j){
          double d=X(i,l) - X(j,l);
          matx(i,j) =  pow(d,2.0);
        }
      }
      dmat=-0.5*exp(nu_vec(l))*Kxx.cwiseProduct(matx);
      g_nu(l) = (invSigma*dmat).trace() - (cninvt.transpose())*dmat*cninvt;
    }
    grad << g_sigma_f,g_sigma_y,g_bias,g_nu;
    grad = 0.5*grad;
    memcpy(resultado_grad,grad.data(),grad.size()*sizeof(double));
  }
  
  void gp_hmc(int* niter,int* leapfrogData,int* burninData,double* KxxData,int* Kxxnrow,int* Kxxncol,double* XData,int* Xnrow,int* Xncol,double* YData,int* Ynrow,int* Yncol,double* sigma_f,double* sigma_y,double* bias,double* nu,double* resultado_hmc,int* resultado_rate){
    int param_dim=(3+*Xncol);
    Map<VectorXd>nu_vec(nu,*Xncol);
    MatrixXd samples = MatrixXd::Zero(*niter+1,param_dim);
    VectorXd p0=VectorXd::Zero(param_dim);
    VectorXd x0=VectorXd::Zero(param_dim);      
    VectorXd pstar=VectorXd::Zero(param_dim);
    VectorXd pstar_n=VectorXd::Zero(param_dim);
    VectorXd xstar=VectorXd::Zero(param_dim);    
    VectorXd leap_grad=VectorXd::Zero(param_dim);
    //double U0, UStar, K0, KStar, alpha, uniforme;
    default_random_engine generator;
    uniform_real_distribution<double> runif(0.0,1.0);
    normal_distribution<double> rnormal(0.0,1.0);
    bernoulli_distribution rbernoulli(0.25);
    double lambda;
    double aleatorio = rnormal(generator);            
    if(aleatorio < 0.5)
      lambda = -1;    
    else
      lambda = 1;
    double delta = lambda*0.02*(1.0+0.1*runif(generator));
    
    /*for(i=2;i<(*niter);i++){     

      for(int p=0;p<(*length_init_par);p++)
        p0(p) = rnormal(generator);

      x0 = init_par;
      gp_grad(XData,XDatanrow,XDatancol,YData,YDatanrow,YDatancol,KxxData,Kxxnrow,Kxxncol,x0(1),x0(2),x0.tail(data_dim).data(),&resultado_grad[0]);
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
            Map<VectorXd>grad2(&resultado_grad[0],param_dim);
            leap_grad = grad2;
          }              
          pstar.head(3) = pstar.head(3) - delta*leap_grad.head(3) ;
          xstar.head(3)  = xstar.head(3)  + delta*pstar.head(3) ;
      }
      cout << "x_star:" << xstar.transpose() << endl;
          
      gp_grad(KxxData,Kxxnrow,Kxxncol,XData,XDatanrow,XDatancol,invSigmaData,invSigmanrow,invSigmancol,trcinvData,cninvtData,cninvtnrow,&xstar(1),&xstar(2),xstar.tail(data_dim).data(),&resultado_grad[0]);  
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
    */
    //memcpy(resultado_hmc,samples.data(),samples.size()*sizeof(double));
    //resultado_rate[0] = rate;  
  }
}
