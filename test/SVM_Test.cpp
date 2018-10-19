#include "svm/svm.hpp"
#include <iostream>

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

int main(){
	Eigen::MatrixXd x;
	x.resize(1, 10);
	x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	Eigen::VectorXd y;
	y.resize(10, 1);
	y << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

	svr svr_instance(x, y, KernelType::linear, 10, 0.01);
	svr_instance.train();

	for(int i = 0; i < 20; i++){
		Eigen::VectorXd x;
		x.resize(1);
		x << i;
		cout << "predict: " << i << " , value: " << svr_instance.predict(x) << endl;
	}
}