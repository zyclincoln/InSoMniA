#include <iostream>
#include <memory>
#include <cassert>
#include "Eigen/Core"
#include "optimizer/optimizer.h"
#include "utility/function_utility.h"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

int main(){
	MatrixXd obj_hessian;
	obj_hessian.resize(2, 2);
	obj_hessian << 2, 0, 0, 2;

	VectorXd obj_gradient;
	obj_gradient.resize(2);
	obj_gradient << -2, -5;

	double obj_constant = 7.25;

	shared_ptr<func<soc> > objectFunction = 
		build_soc_function(obj_hessian, obj_gradient, obj_constant, 2);

	vector<shared_ptr<func<foc>>> ieqc;
	vector<shared_ptr<func<foc>>> eqc;

	Eigen::MatrixXd parameters;
	parameters.resize(5, 2);
	parameters << 1, -2,
				  -1, -2,
				  -1, 2,
				  1, 0,
				  0, 1;
	Eigen::VectorXd constants;
	constants.resize(5);
	constants << 2, 6, 2, 0, 0;
	build_foc_function_from_matrix(parameters, constants, 2, ieqc);

	Eigen::VectorXd point;
	point.resize(2);
	point << 1, 1;

	for(int i = 0; i < ieqc.size(); i++){
		Eigen::VectorXd gradient;
		double value;
		ieqc[i]->value(point, value);
		ieqc[i]->gradient(point, gradient);
	}

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

	assert(fabs(endValue - 0.8) < 1e-10);
}