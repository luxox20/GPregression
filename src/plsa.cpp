/**
 * @file plsa.cpp
 * @brief plsa 
 * @author Guillermo Becerra
 */
#include "plsa.h"

MatrixXd PLSA::logeps(MatrixXd mat){
	for(int i=0;i<mat.rows();i++)
		for(int j=0;j<mat.cols();j++)
			mat(i,j)=log(mat(i,j)+std::numeric_limits<double>::epsilon());
	return mat;
}
void PLSA::normalize(MatrixXd& mat){

	MatrixXd row = mat.colwise().sum();
	for(int i=0;i<mat.rows();i++)
		mat.row(i)=mat.row(i).cwiseQuotient(row);
	
}
void PLSA::normalize(map<int,MatrixXd>& matd,MatrixXd& mat){ 
	for(unsigned int i=0;i<matd.size();i++)
		matd[i]=matd[i].cwiseQuotient(mat);
}

double PLSA::max_error_coeff(const MatrixXd& mat,const MatrixXd& mat2){
	return (mat - mat2).cwiseAbs().maxCoeff(); 
}
double PLSA::mean_error(const MatrixXd& mat,const MatrixXd& mat2){
	return (mat - mat2).cwiseAbs().sum()/(mat.cols()*mat.rows()); 
}

void PLSA::init(int X,int Y,int Z){
	srand((unsigned int) time(0));
	tpz=VectorXd::Random(Z);
	tpxgz=MatrixXd::Random(X,Z);
	tpygz=MatrixXd::Random(Y,Z);
	tpz=tpz.cwiseAbs();
	tpz=tpz/tpz.sum();
	tpxgz=tpxgz.cwiseAbs();
	tpygz=tpygz.cwiseAbs();
	normalize(tpxgz);
	normalize(tpygz);
}

PLSA::PLSA(Map<MatrixXd>& pxy,int Z,int maxIter,double tol){
	plsa(pxy,Z,maxIter,tol);
}

void PLSA::plsa(Map<MatrixXd>& pxy,int Z,int maxIter,double tol){
	int X,Y;
	double loglikehood=0;
	map<int,MatrixXd> qzgxy;
	X=pxy.rows();
	Y=pxy.cols();
	init(X,Y,Z);
	for(int i=0;i<maxIter;i++){

		tpxy = MatrixXd::Zero(X,Y);
		
		for(int z=0;z<Z;z++)
			tpxy+=(tpxgz.col(z)*tpygz.col(z).transpose())*tpz(z);
		
		double loglikehood_ante=loglikehood;
		loglikehood=logeps(tpxy).cwiseProduct(pxy).sum();
		//E-step
		{
			MatrixXd sum_qzgxy; 
			sum_qzgxy=MatrixXd::Zero(X,Y);
			for(int z=0;z<Z;z++){
				qzgxy[z]=(((tpxgz.col(z)*tpygz.col(z).transpose())*tpz(z)).array()+std::numeric_limits<double>::epsilon()).matrix();
				qzgxy[z]=qzgxy[z]*(1.0/(qzgxy[z].sum()));
				sum_qzgxy+=qzgxy[z];
			}

			normalize(qzgxy,sum_qzgxy);	
		}
		//M-step
		for(int z=0;z<Z;z++){
			tpxgz.col(z)=(qzgxy[z].cwiseProduct(pxy)).rowwise().sum();
			tpygz.col(z)=(qzgxy[z].cwiseProduct(pxy)).colwise().sum();
		}
		qzgxy.clear();
		tpz= tpxgz.colwise().sum();
		normalize(tpxgz);
		normalize(tpygz);
		if(i>0 && loglikehood_ante-loglikehood<tol)
			break;					
	}
}


MatrixXd PLSA::test(Map<MatrixXd>& xtest, int iter,int tol){

	map<int,MatrixXd> qzgxy;
	MatrixXd ttpxgz,ttpxy;
	int Z=tpxgz.cols();
	MatrixXd sum_qzgxy;
	ttpxgz=MatrixXd::Random(xtest.rows(),Z);
	ttpxgz=ttpxgz.cwiseAbs();
	normalize(ttpxgz);

	for(int i=0;i<iter;i++){
		sum_qzgxy=MatrixXd::Zero(xtest.rows(),xtest.cols());
		for(int z=0;z<Z;z++){
			qzgxy[z]=(((ttpxgz.col(z)*tpygz.col(z).transpose())*tpz(z)).array()+std::numeric_limits<double>::epsilon()).matrix();
			qzgxy[z]=qzgxy[z]*(1.0/(qzgxy[z].sum()));
			sum_qzgxy+=qzgxy[z];
		}
		normalize(qzgxy,sum_qzgxy);
		for(int z=0;z<Z;z++){
			ttpxgz.col(z)=(qzgxy[z].cwiseProduct(xtest)).rowwise().sum();
		}
		normalize(ttpxgz);
		if(mean_error(ttpxgz,tpxgz)<tol)
			break;
	}
	qzgxy.clear();
	return ttpxgz;
}

//MatrixXd PLSA::pseudo_inverse(const Ref<const MatrixXd> &mat){
	//if(mat.rows()>mat.cols()){
		//return (mat.transpose()*mat).inverse()*mat.transpose();
	//}else{
		//return (mat.transpose()*(mat*mat.transpose()).inverse());
	//}
//}

//MatrixXd PLSA::test_inv(MatrixXd& xtest){

	//return xtest*pseudo_inverse(tpygz.transpose());
	
//}
