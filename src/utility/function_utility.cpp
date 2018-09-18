#include <cassert>
#include "utility/function_utility.h"

using namespace std;
using namespace Eigen;

namespace zyclincoln{
	namespace InSoMniA{

		shared_ptr<func<foc>> build_foc_function(const VectorXd& gradient,
											 	 const double constant,
											  	 const size_t dimension){
			assert(gradient.rows() == dimension);

			return shared_ptr<func<foc>> (
				new std_foc_function(
					[_gradient = gradient, 
					 _constant = constant, 
					 _dimension = dimension]
					(const VectorXd& point, double& value){
						assert(point.rows() == _dimension);
						value = _gradient.dot(point) + _constant;
						return true;
					},
					[_gradient = gradient, 
					 _dimension = dimension]
					(const VectorXd& point, VectorXd& gradient){
						gradient.resize(_dimension);
						gradient = _gradient;
						return true;
					},
					dimension
				));
		}

		bool build_foc_function_from_matrix(const MatrixXd& gradients, 
											const VectorXd& constants,
											const size_t dimension,
											vector<shared_ptr<func<foc>>>& functions){
			assert(gradients.rows() == constants.rows());
			assert(gradients.cols() == dimension);

			for(size_t i = 0; i < gradients.rows(); ++i){
				functions.push_back(build_foc_function(gradients.row(i), 
														constants(i, 0), 
														dimension));
			}

		}

		std::shared_ptr<func<soc>> build_soc_function(const Eigen::MatrixXd& hessian,
													  const Eigen::VectorXd& gradient,
													  const double constant,
													  const size_t dimension){
			assert(hessian.rows() == hessian.cols());
			assert(hessian.rows() == dimension);
			assert(gradient.rows() == dimension);

			return shared_ptr<func<soc>>(
				new std_soc_function(
					[_hessian = hessian,
					 _gradient = gradient,
					 _constant = constant,
					 _dimension = dimension]
					(const VectorXd& point, double& value){
						assert(point.rows() == _dimension);
						value = 0.5 * point.transpose() * _hessian * point;
						value += _gradient.dot(point) + _constant;
						return true;
					},
					[_hessian = hessian,
					 _gradient = gradient,
					 _dimension = dimension]
					(const VectorXd& point, VectorXd& gradient){
						assert(point.rows() == _dimension);
						gradient.resize(_dimension);
						gradient = _hessian * point + _gradient;
						return true;
					},
					[_hessian = hessian,
					 _dimension = dimension]
					(const VectorXd& point, MatrixXd& hessian){
						hessian.resize(_dimension, _dimension);
						hessian = _hessian;
						return true;
					},
					dimension
				)
			);
		}
	}
}