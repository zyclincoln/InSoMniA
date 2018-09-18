#ifndef _FUNCTION_UTILITY_H_
#define _FUNCTION_UTILITY_H_

#include <memory>
#include <vector>
#include "Eigen/Core"
#include "func/func.h"

namespace zyclincoln{
	namespace InSoMniA{

		std::shared_ptr<func<foc>> build_foc_function(const Eigen::VectorXd& gradient,
										 			  const double constant,
										 			  const size_t dimension);

		bool build_foc_function_from_matrix(const Eigen::MatrixXd& gradients, 
											const Eigen::VectorXd& constants,
											const size_t dimension,
											std::vector<std::shared_ptr<func<foc>>>& functions);
	
		std::shared_ptr<func<soc>> build_soc_function(const Eigen::MatrixXd& hessian,
													  const Eigen::VectorXd& gradient,
													  const double constant,
													  const size_t dimension);
	}
}

#endif