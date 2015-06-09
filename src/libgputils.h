#ifdef __cplusplus
extern "C"
#endif
void squared_exponential (double* Xdata, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* nu, int* dim, double* result);
void gp_predict (double* Kxsdata, int* Kxsnrow, int* Kxsncol,double*  Kxxdata, int* Kxxnrow, int* Kxxncol, double*  Ydata, int* Ynrow,int* Yncol, double* MeanPred);
void gp_predict_var (double* Kssdata, int* Kssnrow, int* Kssncol, double* Kxsdata, int* Kxsnrow, int* Kxsncol,double*  Kxxdata, int* Kxxnrow, int* Kxxncol, double* VarPred);
void squared_exponential_geo (double* Xdata, int* Xnrow, int* Xncol,double* Ydata, int* Ynrow, int* Yncol,double* nu, int* dim, double* result);
void log_likelihood (double* Kxxdata, int* Kxxnrow, int* Kxxncol,double* Ydata, int* Ynrow, double* result);


