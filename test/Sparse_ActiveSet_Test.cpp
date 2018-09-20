#include "Eigen/Core"
#include "optimizer/optimizer.h"
#include "utility/function_utility.h"
#include "gtest/gtest.h"

using namespace zyclincoln::InSoMniA;
using namespace Eigen;
using namespace std;

namespace {
	class Sparse_ActiveSet_Test : public ::testing::Test{
	protected:
		Sparse_ActiveSet_Test(){}
		~Sparse_ActiveSet_Test(){}

		virtual void SetUp(){}
		virtual void TearDown(){}
	};

	TEST_F(Sparse_ActiveSet_Test, TestActiveSet){
		shared_ptr<func<soc>> object_function;
		{
			vector<Triplet<double>> hessian;
			vector<Triplet<double>> gradient;

			hessian.push_back(Triplet<double>(0, 0, 2));
			hessian.push_back(Triplet<double>(1, 1, 2));

			gradient.push_back(Triplet<double>(0, 0, -2));
			gradient.push_back(Triplet<double>(1, 0, -5));

			double constant = 7.25;

			object_function = shared_ptr<func<soc>>(new std_soc_func(hessian, gradient, constant, 2));
		}

		vector<shared_ptr<func<foc>>> ieqc;
		vector<shared_ptr<func<foc>>> eqc;
		{
			vector<Triplet<double>> gradient;
			gradient.push_back(Triplet<double>(0, 0, 1));
			gradient.push_back(Triplet<double>(1, 0, -2));
			double constant = 2;
			ieqc.push_back(shared_ptr<func<foc>>(new std_foc_func(gradient, constant, 2)));

			gradient.clear();
			gradient.push_back(Triplet<double>(0, 0, -1));
			gradient.push_back(Triplet<double>(1, 0, -2));
			constant = 6;
			ieqc.push_back(shared_ptr<func<foc>>(new std_foc_func(gradient, constant, 2)));

			gradient.clear();
			gradient.push_back(Triplet<double>(0, 0, -1));
			gradient.push_back(Triplet<double>(1, 0, 2));
			constant = 2;
			ieqc.push_back(shared_ptr<func<foc>>(new std_foc_func(gradient, constant, 2)));			
		
			gradient.clear();
			gradient.push_back(Triplet<double>(0, 0, 1));
			gradient.push_back(Triplet<double>(1, 0, 0));
			constant = 0;
			ieqc.push_back(shared_ptr<func<foc>>(new std_foc_func(gradient, constant, 2)));

			gradient.clear();
			gradient.push_back(Triplet<double>(0, 0, 0));
			gradient.push_back(Triplet<double>(1, 0, 1));
			constant = 0;
			ieqc.push_back(shared_ptr<func<foc>>(new std_foc_func(gradient, constant, 2)));
		}

		optimizer<func<soc>> optimizer(object_function, eqc, ieqc);
		VectorXd initPoint;
		initPoint.resize(2);
		initPoint << 2, 0;
		optimizer.initialize(initPoint);

		VectorXd endPoint;
		double endValue;

		endPoint.resize(2);
		optimizer.solve_active_set(endPoint, endValue, 1e-10, 1000, true);

		cerr << "end point: " << endPoint.transpose() << endl;
		cerr << "end value: " << endValue << endl;
	
		ASSERT_LE(fabs(endValue - 0.8), 1e-10);	
	}
}

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc, argv);
 	return RUN_ALL_TESTS();
}