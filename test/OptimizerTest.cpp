#include "optimizer/optimizer.h"
#include "Eigen/Core"
#include "memory"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

class TestFOC : public func<foc>{
public:
	bool value(const Eigen::VectorXd& point, double& value){
		return false;
	}
	bool grediant(const Eigen::VectorXd& point, Eigen::VectorXd& grediant){
		return false;
	}
};

int main(){

	shared_ptr<func<foc> > objectFunction(new TestFOC());
	optimizer<func<foc> > optimizer(objectFunction);

	VectorXd initPoint;
	initPoint.resize(2);
	initPoint.setZero();
	optimizer.init_point(initPoint);

	VectorXd endPoint;
	double endValue;
	endPoint.resize(2);
	endPoint.setZero();
	optimizer.solve(endPoint, endValue, 1e-10, 100);

	return 0;
}