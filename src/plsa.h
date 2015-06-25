/**
 * @file plsa.hpp
 * @brief plsa 
 * @author Guillermo Becerra
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <Eigen/Dense>
#include <map>
#include <limits>

using namespace std;
using namespace Eigen;

class PLSA{
	public:
		PLSA(){};
		PLSA(Map<MatrixXd>& pxy,int Z,int maxIter,double tol);
		void plsa(Map<MatrixXd>& pxy,int Z,int maxIter,double tol);
		MatrixXd get_pxgz(void){return tpxgz;};
		MatrixXd get_pygz(void){return tpygz;};
		MatrixXd get_tpxy(void){return tpxy;};
		VectorXd get_pz(void){return tpz;};
		MatrixXd test(Map<MatrixXd>& xtest, int iter,int tol);
		double max_error_coeff(const MatrixXd& mat,const MatrixXd& mat2);
		double mean_error(const MatrixXd& mat,const MatrixXd& mat2);	
	private:
		MatrixXd tpxgz;
		MatrixXd tpygz;
		MatrixXd tpxy;
		VectorXd tpz;
		MatrixXd logeps(MatrixXd mat);
		void normalize(map<int,MatrixXd>& matd,MatrixXd& mat);
		void normalize(MatrixXd& mat);
		void init(int X,int Y,int Z);
};
