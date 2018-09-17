#include <iostream>
#include <memory>
#include <cassert>
#include "optimizer/optimizer.h"
#include "Eigen/Core"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

class TestFunction : public func<soc>{
	Eigen::MatrixXd _hessian;
	Eigen::VectorXd _g;

public:
	TestFunction(){
		_hessian.resize(2, 2);
		_hessian << 2, 0, 0, 2;
		_g.resize(2);
		_g << -2, -5;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);

		value = 0.5 * point.transpose() * _hessian * point;
		value += _g.transpose() * point + 7.25;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		assert(point.rows() == 2);
		gradient.resize(2);

		gradient = _hessian * point + _g;
	}

	bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
		hessian.resize(2, 2);
		hessian = _hessian;
	}

	size_t dimension(){return 2;}
};

class C1 : public func<foc>{
	Eigen::VectorXd _parameter;

public:
	C1(){
		_parameter.resize(2);
		_parameter << 1, -2;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		value = _parameter.dot(point) + 2;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient.resize(2);
		gradient = _parameter;
	}

	size_t dimension(){return 2;}
};

class C2 : public func<foc>{
	Eigen::VectorXd _parameter;

public:
	C2(){
		_parameter.resize(2);
		_parameter << -1, -2;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		value = _parameter.dot(point) + 6;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient.resize(2);
		gradient = _parameter;
	}

	size_t dimension(){return 2;}
};

class C3 : public func<foc>{
	Eigen::VectorXd _parameter;

public:
	C3(){
		_parameter.resize(2);
		_parameter << -1, 2;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		value = _parameter.dot(point) + 2;
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient.resize(2);
		gradient = _parameter;
	}

	size_t dimension(){return 2;}
};

class C4 : public func<foc>{
	Eigen::VectorXd _parameter;

public:
	C4(){
		_parameter.resize(2);
		_parameter << 1, 0;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		value = _parameter.dot(point);
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient.resize(2);
		gradient = _parameter;
	}

	size_t dimension(){return 2;}
};

class C5 : public func<foc>{
	Eigen::VectorXd _parameter;

public:
	C5(){
		_parameter.resize(2);
		_parameter << 0, 1;
	}

	bool value(const Eigen::VectorXd& point, double& value){
		assert(point.rows() == 2);
		value = _parameter.dot(point);
	}

	bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
		gradient.resize(2);
		gradient = _parameter;
	}

	size_t dimension(){return 2;}
};


int main(){
	shared_ptr<func<soc> > objectFunction(new TestFunction());

	vector<shared_ptr<func<foc>>> ieqc;
	vector<shared_ptr<func<foc>>> eqc;

	ieqc.push_back(shared_ptr<func<foc>>(new C1()));
	ieqc.push_back(shared_ptr<func<foc>>(new C2()));
	ieqc.push_back(shared_ptr<func<foc>>(new C3()));
	ieqc.push_back(shared_ptr<func<foc>>(new C4()));
	ieqc.push_back(shared_ptr<func<foc>>(new C5()));

	optimizer<func<soc> > optimizer(objectFunction, eqc, ieqc);

	VectorXd initPoint;
	initPoint.resize(2);
	initPoint << 2, 0;

	optimizer.initialize(initPoint);

	VectorXd endPoint;
	double endValue;

	endPoint.resize(2);
	optimizer.solve_active_set(endPoint, endValue, 1e-10, 1000);

	cerr << "end point: " << endPoint.transpose() << endl;
	cerr << "end value: " << endValue << endl;
}