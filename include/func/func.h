#ifndef _FUNC_H_
#define _FUNC_H_

#include "Eigen/Core"

namespace zyclincoln{
	namespace InSoMniA{

		template<typename FT>
		class func;

		class foc{};
		class soc{};

		template<>
		class func<foc>{
		public:
			virtual bool value(const Eigen::VectorXd& point, double& value) = 0;
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& grediant) = 0;
			typedef foc FUNCTION_TYPE;
		};

		template<>
		class func<soc>{
		public:
			virtual bool value(const Eigen::VectorXd& point, double& value) = 0;
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& grediant) = 0;
			virtual bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian) = 0;
			typedef soc FUNCTION_TYPE;
		};
	}
}

#endif