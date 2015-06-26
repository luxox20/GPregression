#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <random> 
#include <limits> 
#include "plsa.h"

using namespace Eigen;
using namespace std;

extern "C"{
  #include <math.h>
  #include <R.h>

   void plsa(double* _pxy,int* Xnrow,int* Xncol,int* Z,int* maxIter,double* tol,double* tpxy,double* pygz,double* pxgz,double* pz){
    PLSA p;
    Map<MatrixXd>pxy(_pxy,*Xnrow,*Xncol);
    p.plsa(pxy,*Z,*maxIter,*tol);
    memcpy(tpxy,p.get_tpxy().data(),p.get_tpxy().rows()*p.get_tpxy().cols()*sizeof(double));
    memcpy(pxgz,p.get_pxgz().data(),p.get_pxgz().rows()*p.get_pxgz().cols()*sizeof(double));
    memcpy(pygz,p.get_pygz().data(),p.get_pygz().rows()*p.get_pygz().cols()*sizeof(double));
    memcpy(pz,p.get_pz().data(),p.get_pz().size()*sizeof(double));
  }
  //no se puede hacer sobrecarga
  void plsa_with_test(double* _pxy,int* Xnrow,int* Xncol,int* Z,int* maxIter,double* tol,double* tpxy,double* pygz,double* pxgz,double* pz,double *test,int *testXnRows,int *testXnCols,double *test_result){
    PLSA p;
    Map<MatrixXd>pxy(_pxy,*Xnrow,*Xncol);
    Map<MatrixXd>xtest(test,*testXnRows,*testXnCols);
    MatrixXd test_r;
    p.plsa(pxy,*Z,*maxIter,*tol);
    test_r=p.test(xtest,*maxIter,*tol);
    memcpy(tpxy,p.get_tpxy().data(),p.get_tpxy().rows()*p.get_tpxy().cols()*sizeof(double));
    memcpy(pxgz,p.get_pxgz().data(),p.get_pxgz().rows()*p.get_pxgz().cols()*sizeof(double));
    memcpy(pygz,p.get_pygz().data(),p.get_pygz().rows()*p.get_pygz().cols()*sizeof(double));
    memcpy(pz,p.get_pz().data(),p.get_pz().size()*sizeof(double));
    memcpy(test_result,test_r.data(),test_r.rows()*test_r.cols()*sizeof(double));
  }

  void squared_exponential_train(double* XData, int* Xnrow, int* Xncol,double* sigma_f, double* nu, int* dim, double* result){
    Map<MatrixXd>X(XData,*Xnrow,*Xncol);
    Map<VectorXd>nu_vector(nu,*Xncol);
    MatrixXd Cov;
    Cov.resize(*Xnrow,*Xnrow);
    for(int i=0; i < *Xnrow; ++i){      
      for(int j=i; j < *Xnrow; ++j){
        VectorXd d=(X.row(i)-X.row(j));
        double S=(nu_vector.array().exp()*d.array().pow(2)).sum();
        Cov(i,j) =  exp(*sigma_f - 0.5 * S);
      }
    }
    memcpy(result,Cov.data(),Cov.size() * sizeof(double));
  } 

  void squared_exponential(double* XData, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* sigma_f, double* nu, int* dim, double* result){
	  Map<MatrixXd>X(XData,*Xnrow,*Xncol);
	  Map<MatrixXd>Y(Ydata,*Ynrow, *Yncol);
    Map<VectorXd>nu_vector(nu,*Xncol);
    MatrixXd Cov;
    Cov.resize(*Xnrow,*Ynrow);
		for(int i=0; i < *Xnrow; ++i){			
			for(int j=0; j < *Ynrow; ++j){
        double S=0;
        for(int k=0;k<*Xncol;++k){
				  double d=(X(i,k)-Y(j,k));
          S+=exp(nu[k])*d;
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
  
  void gp_hmc(int* niter,int* leapfrog,int* burnin,double* KxxData,int* Kxxnrow,int* Kxxncol,double* XData,int* Xnrow,int* Xncol,double* YData,int* Ynrow,int* Yncol,double* init_par,int* param_dim,double* resultado_hmc,int* resultado_rate){
    Map<VectorXd>x0(init_par,*param_dim);
    MatrixXd samples = MatrixXd::Zero(*niter,*param_dim);
    default_random_engine generator;
    uniform_real_distribution<double> runif(0.0,1.0);
    normal_distribution<double> rnormal(0.0,1.0);
    bernoulli_distribution rbernoulli(0.25);
    double lambda;
    int rate=0;          
    if(rnormal(generator) < 0.5)
      lambda = -1;    
    else
      lambda = 1;
    samples.row(0)=x0;
    double resultado_grad[*param_dim];
    for(int i=1;i<(*niter);i++){     
      VectorXd p0=VectorXd::Zero(*param_dim);
      VectorXd pstar,xstar;
      double ll_0,ll_star;
      for(int p=0;p<(*param_dim);p++) {
        p0(p) = rnormal(generator);
      }
      squared_exponential(XData,Xnrow,Xncol,XData,Xnrow,Xncol,&x0(0),x0.tail(*Xncol).data(),Xncol,KxxData);
      log_likelihood(KxxData,Kxxnrow,Kxxncol,YData,Ynrow,&x0(0),&x0(1),&x0(2),&ll_0);
      double K0 = p0.cwiseProduct(p0).sum()*0.5;
      gp_grad(XData,Xnrow,Xncol,YData,Ynrow,Yncol,KxxData,Kxxnrow,Kxxncol,&x0(0),&x0(1),&x0(2),x0.tail(*Xncol).data(),&resultado_grad[0]);
      Map<VectorXd>grad(&resultado_grad[0],*param_dim);
      double delta = lambda*0.02*(1.0+0.1*runif(generator));
      //double delta=0.3;
      pstar = p0 - (delta/2)*grad;
      xstar = x0 + delta*pstar;
      for(int j=1;j<=(*leapfrog-1);j++){
          squared_exponential(XData,Xnrow,Xncol,XData,Xnrow,Xncol,&xstar(0),xstar.tail(*Xncol).data(),Xncol,KxxData);    
          gp_grad(XData,Xnrow,Xncol,YData,Ynrow,Yncol,KxxData,Kxxnrow,Kxxncol,&xstar(0),&xstar(1),&xstar(2),xstar.tail(*Xncol).data(),&resultado_grad[0]);         
          Map<VectorXd>leap_grad(&resultado_grad[0],*param_dim);           
          pstar.noalias() -=  delta*leap_grad;
          xstar.noalias() +=  delta*pstar;
      }             
      squared_exponential(XData,Xnrow,Xncol,XData,Xnrow,Xncol,&xstar(0),xstar.tail(*Xncol).data(),Xncol,KxxData); 
      log_likelihood(KxxData,Kxxnrow,Kxxncol,YData,Ynrow,&xstar(0),&xstar(1),&xstar(2),&ll_star);
      double KStar = pstar.cwiseProduct(pstar).sum()*0.5;
      double alpha = min(1.0,exp((ll_star+KStar)-(ll_0+K0)));     
      if(runif(generator)< alpha){
          samples.row(i) = xstar.transpose();
          x0=xstar;
          rate++;
      }
      else{
         samples.row(i) = x0.transpose();
      }
    }
    //samples.resize(rate,*param_dim); 
    memcpy(resultado_hmc,samples.data(),samples.size()*sizeof(double));
    resultado_rate[0] = rate;  
  }
}
