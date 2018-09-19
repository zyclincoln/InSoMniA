#include "matrix_solver/KKT_solver.h"
#include "utility/matrix_utility.h"
#include <vector>
#include <iostream>
#include "Eigen/SparseLU" 

using namespace Eigen;
using namespace std;

namespace zyclincoln{
	namespace InSoMniA{

		bool trival_kkt_solver(const MatrixXd& hessian,
							   const MatrixXd& A,
							   const VectorXd& g,
							   const VectorXd& h,
							   VectorXd& x){
			assert(hessian.cols() == hessian.rows());
			assert(hessian.cols() == A.cols());
			assert(g.rows() == hessian.rows());
			assert(h.rows() == A.rows());

			MatrixXd kkt_matrix;
			kkt_matrix.resize(hessian.rows() + A.rows(), hessian.cols() + A.rows());
			kkt_matrix.setZero();

			kkt_matrix.block(0, 0, hessian.rows(), hessian.cols()) = hessian;
			kkt_matrix.block(hessian.rows(), 0, A.rows(), A.cols()) = A;
			kkt_matrix.block(0, hessian.cols(), A.cols(), A.rows()) = A.transpose();

			VectorXd gh;
			vector<Eigen::VectorXd> candidate {g, h};
			merge_vector(candidate, gh);

			// gh.resize(g.rows() + h.rows());

			// gh.segment(0, g.rows()) = g;
			// gh.segment(g.rows(), h.rows()) = h;

			x.resize(g.rows() + h.rows());
			x = kkt_matrix.inverse() * gh;

			return true;
		}

		bool sparse_factor_kkt_solver(const std::vector<Eigen::Triplet<double>>& hessian,
									  const std::vector<Eigen::Triplet<double>>& A,
									  const Eigen::VectorXd& g,
									  const Eigen::VectorXd& h,
									  Eigen::VectorXd& x){
			
			SparseMatrix<double, ColMajor> kkt_matrix;
			kkt_matrix.resize(g.rows() + h.rows(), g.rows() + h.rows());

			// here, reserve memory space for kkt matrix, which may need more info from outside
			kkt_matrix.reserve(Eigen::VectorXi::Constant(g.rows() + h.rows(), 3));

			std::vector<Eigen::Triplet<double>> kkt_triplet;
			for(auto i = hessian.cbegin(); i != hessian.cend(); i++){
				kkt_triplet.push_back(*i);
			}

			int shift = g.rows();
			for(auto i = A.cbegin(); i != A.cend(); i++){
				kkt_triplet.push_back(Eigen::Triplet<double>(i->row()+shift, i->col(), i->value()));
				kkt_triplet.push_back(Eigen::Triplet<double>(i->col()+shift, i->row(), i->value()));
			}

			kkt_matrix.setFromTriplets(kkt_triplet.begin(), kkt_triplet.end());
			kkt_matrix.makeCompressed();

			VectorXd gh;
			vector<Eigen::VectorXd> candidate {g, h};
			merge_vector(candidate, gh);

			SparseLU<SparseMatrix<double>, COLAMDOrdering<int> > solver;
			solver.analyzePattern(kkt_matrix);
			solver.factorize(kkt_matrix);

			x.resize(gh.rows());
			x = solver.solve(gh);

			return true;
		}

	}
}