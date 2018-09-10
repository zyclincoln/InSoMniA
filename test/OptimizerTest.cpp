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

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		assert(point.rows() == 2);
		gradient.resize(2);
		const double &x = point(0, 0);
		const double &y = point(1, 0);
		gradient(0, 0) = 200 * (y - x*x)*(-2 * x) - 2*(1 - x);
		gradient(1, 0) = 200 * (y - x*x);
		return true;
	}
};

class Rosenbrock2 : public func<soc>{
public:
	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		const double &x = point(0, 0);
		const double &y = point(1, 0);

		value = 100 * (y - x*x)*(y - x*x) + (1 - x)*(1 - x);
		return true;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		assert(point.rows() == 2);
		gradient.resize(2);
		const double &x = point(0, 0);
		const double &y = point(1, 0);
		gradient(0, 0) = 200 * (y - x*x)*(-2 * x) - 2*(1 - x);
		gradient(1, 0) = 200 * (y - x*x);
		return true;
	}

	bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
		assert(point.rows() == 2);
		hessian.resize(2, 2);
		const double&x = point(0, 0);
		const double&y = point(1, 0);
		hessian(0, 0) = 400 * (3*x*x - y) + 2;
		hessian(1, 0) = -400 * x;
		hessian(0, 1) = hessian(1, 0);
		hessian(1, 1) = 200;
		return true;
	}
};

int main(){

	shared_ptr<func<foc> > objectFunction(new Rosenbrock());
	shared_ptr<func<soc> > objectFunction2(new Rosenbrock2());
	// optimizer<func<foc> > optimizer(objectFunction);
	optimizer<func<soc> > optimizer(objectFunction2);

	VectorXd initPoint;
	initPoint.resize(2);
	initPoint << 1.2, 1.2;
	// initPoint << -1.2, 1;
	optimizer.initialize(initPoint);

	VectorXd endPoint;
	double endValue;
	endPoint.resize(2);
	endPoint.setZero();

	optimizer.solve(endPoint, endValue, 1e-20, 20000);

	cerr << endPoint.transpose() << endl;
	cerr << endValue << endl;

	return 0;
}