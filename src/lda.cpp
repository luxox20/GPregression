#include "lda.hpp"

LDA::LDA(MatrixXi& _documents,int _V){
	V=_V;
	documents=&_documents;
	dispcol=0;
	numstats=0;
	//default values
	THIN_INTERVAL= 20;
	BURN_IN= 100;
	ITERATIONS= 1000;
	SAMPLE_LAG= 0;
	
}
LDA::~LDA(){
	thetasum.resize(0,0);
	phisum.resize(0,0);
	nw.resize(0,0);
	nd.resize(0,0);
	z.resize(0,0);
	nwsum.resize(0);
	ndsum.resize(0);
	documents=NULL;
}

int LDA::documentsSize(void){
	return (*documents).rows();
}


void LDA::initialState(int K){
	int M=documentsSize(),N=(*documents).cols();
	nw = MatrixXi::Zero(V,K);
	nd = MatrixXi::Zero(M,K);
	nwsum = VectorXi::Zero(K);
	ndsum = VectorXi(M);
	
	z=(MatrixXd::Random(M,N).cwiseAbs()*K).cast<int>();
	ndsum.fill(N);
	for(int i=0;i<M;i++){
		for(int j=0;j<N;j++){
			nw((*documents)(i,j),z(i,j))++;
			nd(i,z(i,j))++;
			nwsum(z(i,j))++;
		}
	}

}

int LDA::sampleFullConditional(int m,int n){
	double u,topic=z(m,n);
	VectorXd p(K);
	nw((*documents)(m,n),topic)--;
	nd(m,topic)--;
	nwsum(topic)--;
	ndsum(m)--;
	for(int k=0;k<K;k++)
		p(k)=((nw((*documents)(m,n),k)+beta)/(nwsum(k)+V*beta)) * ((nd(m,k)+alpha)/(ndsum(m)+K*alpha));

	for(int i=1;i<p.size();i++)
		p(i)+=p(i-1);

	
	u = ((double) rand() / (RAND_MAX))*p(K-1);

	for(topic=0;topic<p.size();topic++)
		if(u<p(topic)) break;

	nw((*documents)(m,n),topic)++;
	nd(m,topic)++;
	nwsum(topic)++;
	ndsum(m)++;
	return topic;

}

void LDA::gibbs(int K,double alpha,double beta){

	this->K=K;
	this->alpha=alpha;
	this->beta=beta;

	if(SAMPLE_LAG >0){
		thetasum = MatrixXd(documentsSize(),K);
		phisum= MatrixXd(K,V);
		numstats=0;
	}

	initialState(K);

	for(int i=0;i<ITERATIONS;i++){
		for(int m=0;m<z.rows();m++){
			for(int n=0;n<z.cols();n++){
				z(m,n)=sampleFullConditional(m,n);
			}
		}
		if(i > BURN_IN && (i%THIN_INTERVAL==0)){
			dispcol++;
		}
		if ((i > BURN_IN) && (SAMPLE_LAG > 0) && (i % SAMPLE_LAG == 0)) {
                updateParams();
                if (i % THIN_INTERVAL != 0)
                    dispcol++;
        }
        if (dispcol >= 100) {
            dispcol = 0;
        }
    }	
}

void LDA::updateParams(void){
	for(int m=0;m<documentsSize();m++){
		for(int k=0;k<K;k++){
			thetasum(m,k)+=(nd(m,k)+alpha)/(ndsum(m)+K*alpha);
		}
	}
	for(int k=0;k<K;k++){
		for(int w=0;w<V;w++){
			phisum(k,w)+= (nw(w,k)+beta)/(nwsum(k)+V*beta);
		}
	}
	numstats++;
}

MatrixXd LDA::getTheta(void){
	if(SAMPLE_LAG>0){
		return thetasum*(1.0/numstats);
	}
	else{
		MatrixXd theta(documentsSize(),K);
		for(int m=0;m<documentsSize();m++){
			// theta.row(m)=((nd.row(m).cast<double>().array()+alpha)/(ndsum.cast<double>().array()+K*alpha)).matrix();
			for(int k=0;k<K;k++){
				theta(m,k)=(nd(m,k)+alpha)/(ndsum(m)+K*alpha);
			}		
		}		
		return theta;
	}
	
}

MatrixXd LDA::getPhi(void){
	
	if(SAMPLE_LAG >0){
		return phisum*(1.0/numstats);
	}else{
		MatrixXd phi(K,V);
		for(int k=0;k<K;k++)
			for(int w=0;w<V;w++)
				phi(k,w)=(nw(w,k)+beta)/(nwsum(k)+V*beta);
		return phi;
	}
}

void LDA::configure(int ITERATIONS, int BURN_IN,int THIN_INTERVAL,int SAMPLE_LAG){
	this->THIN_INTERVAL=THIN_INTERVAL;
	this->BURN_IN=BURN_IN;
	this->ITERATIONS=ITERATIONS;
	this->SAMPLE_LAG=SAMPLE_LAG;
}

void LDA::set_thin_interval(int THIN_INTERVAL){
	this->THIN_INTERVAL=THIN_INTERVAL;
}
void LDA::set_burn_in(int BURN_IN){
	this->BURN_IN=BURN_IN;
}
void LDA::set_iterations(int ITERATIONS){
	this->ITERATIONS=ITERATIONS;
}
void LDA::set_sample_lag(int SAMPLE_LAG){
	this->SAMPLE_LAG=SAMPLE_LAG;
}


