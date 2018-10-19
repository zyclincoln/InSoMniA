#ifndef _FUNC_H_
#define _FUNC_H_

#include "Eigen/Core"
#include "Eigen/Sparse"
#include <iostream>

namespace zyclincoln{
	namespace InSoMniA{

		template<typename FT>
		class func;

		// first order continous
		class foc{};

		// second order continous
		class soc{};

		template<>
		class func<foc>{
		public:
			virtual bool value(const Eigen::VectorXd& point, double& value) = 0;
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				std::cerr << "[FATAL] dense gradient is not implement" << std::endl;
				assert(false);
				return false;
			}

			virtual bool gradient_sparse(const Eigen::VectorXd& point, 
										 std::vector<Eigen::Triplet<double>>& gradient){
				std::cerr << "[FATAL] sparse gradient is not implement" << std::endl;
				assert(false);
				return false;
			}

			virtual size_t dimension() = 0;
			typedef foc FUNCTION_TYPE;
		};

		template<>
		class func<soc>{
		public:
			virtual bool value(const Eigen::VectorXd& point, double& value) = 0;
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				std::cerr << "[FATAL] dense gradient is not implement" << std::endl;
				assert(false);
				return false;
			}

			virtual bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
				std::cerr << "[FATAL] dense hessian is not implement" << std::endl;
				assert(false);
				return false;
			}
			
			virtual bool gradient_sparse(const Eigen::VectorXd& point,
										 std::vector<Eigen::Triplet<double>>& gradient){
				std::cerr << "[FATAL] sparse gradient is not implement" << std::endl;
				assert(false);
				return false;
			}
			
			virtual bool hessian_sparse(const Eigen::VectorXd& point,
										std::vector<Eigen::Triplet<double>>& hessian){
				std::cerr << "[FATAL] sparse hessian is not implement" << std::endl;
				assert(false);
				return false;
			}

			virtual size_t dimension() = 0;
			typedef soc FUNCTION_TYPE;
		};

		class std_foc_func : public func<foc>{
		private:
			size_t _dimension;
			double _constant;
			std::vector<Eigen::Triplet<double>> _gradient_triplets;
			Eigen::SparseVector<double> _gradient;
		public:
			std_foc_func(
				const std::vector<Eigen::Triplet<double>>& gradient_triplets, 
				const double constant,
				const size_t dimension):
				_gradient_triplets(gradient_triplets), 
				_constant(constant), 
				_dimension(dimension){
				
				_gradient.resize(_dimension);
				for(int i = 0; i < gradient_triplets.size(); i++){
					_gradient.coeffRef(gradient_triplets[i].row()) = gradient_triplets[i].value();
				}
				assert(_gradient.rows() == _dimension);
			}

			bool value(const Eigen::VectorXd& point, double& value){
				assert(point.rows() == _dimension);

				value = _gradient.dot(point) + _constant;
				return true;
			}

			bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				assert(point.rows() == _dimension);
				gradient = _gradient;
				return true;
			}

			size_t dimension() { return _dimension; }

			bool gradient_sparse(const Eigen::VectorXd& point, 
								 std::vector<Eigen::Triplet<double>>& gradient){
				assert(point.rows() == _dimension);

				gradient = _gradient_triplets;
				return true;
			}
		};

		class std_soc_func : public func<soc>{
		private:
			Eigen::SparseMatrix<double> _hessian;
			std::vector<Eigen::Triplet<double>> _hessian_triplets;
			Eigen::SparseVector<double> _gradient;
			std::vector<Eigen::Triplet<double>> _gradient_triplets;

			double _constant;
			size_t _dimension;
		public:
			std_soc_func(
				const std::vector<Eigen::Triplet<double>>& hessian_triplets,
				const std::vector<Eigen::Triplet<double>>& gradient_triplets,
				const double constant,
				const size_t dimension):
				_hessian_triplets(hessian_triplets), 
				_gradient_triplets(gradient_triplets), 
				_constant(constant), _dimension(dimension){

				_hessian.resize(dimension, dimension);
				_gradient.resize(dimension);

				_hessian.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());
				for(int i = 0; i < gradient_triplets.size(); i++)
					_gradient.coeffRef(gradient_triplets[i].row()) = gradient_triplets[i].value();

				assert(_hessian.rows() == _hessian.cols());
				assert(_hessian.cols() == _gradient.rows());
				assert(_gradient.rows() == dimension);
			}

			bool value(const Eigen::VectorXd& point, double& value){
				assert(point.rows() == _dimension);

				value = 0.5 * point.transpose() * _hessian * point;
				value += point.transpose() * _gradient + _constant;
				return true;
			}

			bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				assert(point.rows() == _dimension);
				gradient = _hessian * point + _gradient;
				return true;
			}

			bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
				assert(point.rows() == _dimension);
				hessian = _hessian;
				return true;
			}

			bool gradient_sparse(const Eigen::VectorXd& point, 
								 std::vector<Eigen::Triplet<double>>& gradient){
				assert(point.rows() == _dimension);

				Eigen::VectorXd g;
				g = _hessian * point + _gradient;
				for(int i = 0; i < g.rows(); i++){
					if(g(i) == 0){
						gradient.push_back(Eigen::Triplet<double>(i, 0, g(i)));
					}
				}
				return true;
			}

			bool hessian_sparse(const Eigen::VectorXd& point, 
								std::vector<Eigen::Triplet<double>>& hessian){
				assert(point.rows() == _dimension);
				hessian = _hessian_triplets;
				return true;
			}

			size_t dimension() { return _dimension; }
		};

		class std_foc_function : public func<foc>{
		private:
			std::function<bool(const Eigen::VectorXd&, double&)> _value;
			std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)> _gradient;
			size_t _dimension;
		public:
			std_foc_function(
				const std::function<bool(const Eigen::VectorXd&, double&)>& value,
				const std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)>& gradient,
				size_t dimension):
				_value(value), _gradient(gradient), _dimension(dimension){

			}

			bool value(const Eigen::VectorXd& point, double& value){
				return _value(point, value);
			}

			bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				return _gradient(point, gradient);
			}

			size_t dimension(){
				return _dimension;
			}

		};

		class std_soc_function : public func<soc>{
		private:
			std::function<bool(const Eigen::VectorXd&, double&)> _value;
			std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)> _gradient;
			std::function<bool(const Eigen::VectorXd&, Eigen::MatrixXd&)> _hessian;
			size_t _dimension;

		public:
			std_soc_function(
				const std::function<bool(const Eigen::VectorXd&, double&)>& value,
				const std::function<bool(const Eigen::VectorXd&, Eigen::VectorXd&)>& gradient,
				const std::function<bool(const Eigen::VectorXd&, Eigen::MatrixXd&)>& hessian,
				size_t dimension):
			_value(value), _gradient(gradient), _hessian(hessian), _dimension(dimension){

			}

			bool value(const Eigen::VectorXd& point, double& value){
				return _value(point, value);
			}

			bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient){
				return _gradient(point, gradient);
			}

			bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian){
				return _hessian(point, hessian);
			}

			size_t dimension(){
				return _dimension;
			}
		};
	}
}

#endif