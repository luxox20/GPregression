/**
 * @file mainPLSA.cpp
 * @brief plsa demo 
 * @author Guillermo Becerra
 */

#include <iostream>
#include "plsa.hpp"
#include <string>
#include <cstdlib>

int main(void){
	srand((unsigned int) time(0));
	double prom=0;
	for(int h=0;h<100;h++){

		int X=10, Y=10,Z=4,maxIter=100,tolerance=0.000001;
		MatrixXd pxy(X,Y);
		MatrixXd pxgz(X,Z);
		MatrixXd pygz(Y,Z);
		VectorXd pz(Z);
		PLSA p;

		//generate data
		pz=VectorXd::Random(Z);
		pxgz=MatrixXd::Random(X,Z);
		pygz=MatrixXd::Random(Y,Z);
		
		//normalize
		pz=pz.cwiseAbs();
		pz=pz/pz.sum();
		pxgz=pxgz.cwiseAbs();
		pygz=pygz.cwiseAbs();
		for(int i=0;i<pxgz.rows();i++)
			pxgz.row(i)=pxgz.row(i).cwiseQuotient(pxgz.colwise().sum());
		for(int i=0;i<pygz.rows();i++)
			pygz.row(i)=pygz.row(i).cwiseQuotient(pygz.colwise().sum());

		//build pxy
		pxy = MatrixXd::Zero(X,Y);
		for(int z=0;z<Z;z++){
			pxy+=(pxgz.col(z)*pygz.col(z).transpose())*pz(z);
		}

		// pxy=pxy*1000;
		// cout << pxy << endl;

		//PLSA
		p.plsa(pxy,Z,maxIter,tolerance);

		// cout << "----MAX COEFF ERROR -----" << endl;
		// cout << "reconstruction pxy error "<< endl;
		// cout << p.max_error_coeff(pxy,p.get_tpxy())<<endl;
		// cout << "reconstruction pxgz error "<< endl;
		// cout << p.max_error_coeff(pxgz,p.get_pxgz())<<endl;
		// cout << "reconstruction pxgz error "<< endl;
		// cout << p.max_error_coeff(pygz,p.get_pygz())<<endl;
		// cout << "reconstruction pz error "<< endl;
		// cout << p.max_error_coeff(pz,p.get_pz())<<endl<<endl;

		// cout << "----AVERAGE ERROR-----" << endl;
		// cout << "average pxy error "<< endl;
		// cout << p.mean_error(pxy,p.get_tpxy())<<endl;
		// cout << "average pxgz error "<< endl;
		// cout << p.mean_error(pxgz,p.get_pxgz())<<endl;
		// cout << "average pxgz error "<< endl;
		// cout << p.mean_error(pygz,p.get_pygz())<<endl;
		// cout << "average pz error "<< endl;
		// cout << p.mean_error(pz,p.get_pz())<<endl;

		// cout << "------------reconstruction-----------" << endl;
		// cout << "---average error p.get_pxgz() vs test_reconstruction" << endl;
		// cout << p.mean_error(p.get_pxgz(), p.test(pxy,maxIter,tolerance)) << endl;
		// cout << "---average error p.get_pxgz() vs test_reconstruction with pseudo_inv" << endl;
		// cout << p.mean_error(p.get_pxgz(), p.test_inv(pxy)) << endl;
		// cout << "------------------------------------------" << endl;
		// cout << "---average error original_pxgz vs test_reconstruction" << endl;
		// cout << p.mean_error(pxgz, p.test(pxy,maxIter,tolerance)) << endl;
		// cout << "---average error original_pxgz vs test_reconstruction with pseudo_inv" << endl;
		// cout << p.mean_error(pxgz, p.test_inv(pxy)) << endl;

		prom+=  p.mean_error(p.get_pxgz(), p.test(pxy,maxIter,tolerance));
		cout << h << endl;
		sleep(2);

	}
	cout << prom/100 << endl;
}