#ifndef _MATRIX_UTILITY_H_
#define _MATRIX_UTILITY_H_

#include <vector>
#include "Eigen/Core"

namespace zyclincoln{
	namespace InSoMniA{
		bool merge_vector(const std::vector<Eigen::VectorXd>& candidates,
						  Eigen::VectorXd& result);

	}
}

#endif