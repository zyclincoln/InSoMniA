#ifndef _SVM_HPP_
#define _SVM_HPP_

#include <vector>
#include <set>
#include <Eigen/Core>


namespace zyclincoln{
	namespace InSoMniA{
		enum KernelType{
			linear
		};


		class svr{
		private:
			// column wise
			const Eigen::MatrixXd _x;
			const Eigen::VectorXd _y;
			const KernelType _kernel;
			const double _C, _epsilon;
			Eigen::VectorXd _a;
			size_t _max_iterate_num;
			std::set<size_t> _F_set, _Ac_set;

			double evaluate(const Eigen::VectorXd& x) const;
			double kernel_eval(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const;
		public:
			svr(const Eigen::MatrixXd& x, const Eigen::VectorXd& y, 
				const KernelType kernel, 
				const double C, const double epsilon):
				_x(x), _y(y), _kernel(kernel), _C(C), _epsilon(epsilon){
					_a.resize(x.cols());
					_a.setZero();
					_max_iterate_num = 10000;
			}

			double predict(const Eigen::VectorXd& x) const;
			void train();
		};
	}
}


#endif