#ifndef _KKT_SOLVER_H_
#define _KKT_SOLVER_H_

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

namespace zyclincoln{
	namespace InSoMniA{
		bool trival_kkt_solver(const Eigen::MatrixXd& hessian,
							   const Eigen::MatrixXd& A,
							   const Eigen::VectorXd& g,
							   const Eigen::VectorXd& h,
							   Eigen::VectorXd& x);

		bool sparse_nullspace_kkt_solver(const std::vector<Eigen::Triplet<double>>& hessian,
										 const std::vector<Eigen::Triplet<double>>& A,
										 const Eigen::VectorXd& g,
										 const Eigen::VectorXd& h,
										 Eigen::VectorXd& x);

		bool sparse_factor_kkt_solver(const std::vector<Eigen::Triplet<double>>& hessian,
									  const std::vector<Eigen::Triplet<double>>& A,
									  const Eigen::VectorXd& g,
									  const Eigen::VectorXd& h,
									  Eigen::VectorXd& x);
	}
}

#endif