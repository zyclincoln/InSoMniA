#include <iostream>
#include <memory>
#include <cassert>
#include "optimizer/optimizer.h"
#include "Eigen/Core"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

class TestFunction: public func<soc>{
	Eigen::MatrixXd _hessian;
	Eigen::VectorXd _g;

public:
	TestFunction(){
		_hessian.resize(3, 3);
		_hessian << 6, 2, 1, 
				   2, 5, 2, 
				   1, 2, 4;
				   
		_g.resize(3);
		_g << -8, -3, -3;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 3);

		value = 0.5 * point.transpose() * _hessian * point;
		value += _g.transpose() * point;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		assert(point.rows() == 3);
		gradient = _hessian*point + _g;
	}

	bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
		hessian.resize(3, 3);
		hessian = _hessian;
		return true;
	}

	size_t dimension(){
		return 3;
	}
};

class Constraint1: public func<foc>{
	Eigen::VectorXd _parameter;

public:
	Constraint1(){
		_parameter.resize(3);
		_parameter << 1, 0, 1;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 3);

		value = _parameter.dot(point) - 3; 
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient = _parameter;
	}

	size_t dimension(){
		return 3;
	}
};

class Constraint2: public func<foc>{
	Eigen::VectorXd _parameter;

public:
	Constraint2(){
		_parameter.resize(3);
		_parameter << 0, 1, 1;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 3);
		value = _parameter.dot(point); 
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient = _parameter;
	}

	size_t dimension(){
		return 3;
	}
};


int main(){
	shared_ptr<func<soc> > objectFunction(new TestFunction());

	vector<shared_ptr<func<foc >>> eqc;
	eqc.push_back(shared_ptr<func<foc>>(new Constraint1()));
	eqc.push_back(shared_ptr<func<foc>>(new Constraint2()));

	optimizer<func<soc> > optimizer(objectFunction, eqc);

	VectorXd initPoint;
	initPoint.resize(3);
	initPoint << 0, 0, 0;

	optimizer.initialize(initPoint);

	VectorXd endPoint;
	double endValue;
	endPoint.resize(2);
	optimizer.solve_eqc(endPoint, endValue);

	cerr << endPoint.transpose() << endl;
	cerr << endValue << endl;

	return 0;
}