#include <Eigen/Dense>
#include <map>
#include <cstdlib>

using namespace Eigen;
using namespace std;

class LDA {
	private:
		int V,K,numstats;
		int dispcol;
		double alpha,beta;
		int THIN_INTERVAL, BURN_IN, ITERATIONS, SAMPLE_LAG;
		MatrixXd thetasum,phisum;
		MatrixXi nw,nd,z,*documents;
		VectorXi nwsum,ndsum;
		int sampleFullConditional(int m,int n);
		int documentsSize(void);
		void updateParams(void);	
	public:
		LDA(MatrixXi& _documents,int _V);
		~LDA();
		void gibbs(int K,double alpha,double beta);	
		void initialState(int K);
		MatrixXd getTheta(void);
		MatrixXd getPhi(void);
		void configure(int ITERATIONS, int BURN_IN,int THIN_INTERVAL,int SAMPLE_LAG);
		void set_thin_interval(int THIN_INTERVAL);
		void set_burn_in(int BURN_IN);
		void set_iterations(int ITERATIONS);
		void set_sample_lag(int SAMPLE_LAG);
};
