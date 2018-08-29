#include <iostream>
#include <memory>
#include <cassert>
#include "optimizer/optimizer.h"
#include "Eigen/Core"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

class Rosenbrock : public func<foc>{
public:
	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		const double &x = point(0, 0);
		const double &y = point(1, 0);

		value = 100 * (y - x*x)*(y - x*x) + (1 - x)*(1 - x);
		return true;
	}

	bool grediant(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		assert(point.rows() == 2);
		gradient.resize(2);
		const double &x = point(0, 0);
		const double &y = point(1, 0);
		gradient(0, 0) = 200 * (y - x*x)*(-2 * x) - 2*(1 - x);
		gradient(1, 0) = 200 * (y - x*x);
		return true;
	}
};

int main(){

	shared_ptr<func<foc> > objectFunction(new Rosenbrock());
	optimizer<func<foc> > optimizer(objectFunction);

	VectorXd initPoint;
	initPoint.resize(2);
	initPoint << 1.2, 1.2;
	optimizer.initialize(initPoint);

	VectorXd endPoint;
	double endValue;
	endPoint.resize(2);
	endPoint.setZero();

	optimizer.solve(endPoint, endValue, 1e-20, 10000);

	cerr << endPoint.transpose() << endl;
	cerr << endValue << endl;

	return 0;
}