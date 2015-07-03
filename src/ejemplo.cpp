#include <iostream>
#include <Eigen/Dense>
#include <math.h>
using namespace std;
using namespace Eigen;
int main()
{
   Matrix2f A, b;
   A << 2, -1, -1, 3;
   LLT<Matrix2f> llt(A);
   Matrix2f L = llt.matrixL();
   b << 1, 2, 3, 1;
   cout << "Here is the matrix A:\n" << A << endl;
   cout << "Log determinant:\n" << log(A.determinant()) << endl;
   cout << "The matrix L is now:\n" << L << endl;
   cout << "The matrix L is now:\n" << L*L.transpose() << endl;
   cout << "Log determinant:\n" << 2*L.transpose().diagonal().array().log().sum() << endl;
   
}