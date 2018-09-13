#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <memory>
#include <limits>
#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "func/func.h"
#include "utility/kkt_util.h"
#include "utility/line_search.h"

namespace zyclincoln{

	namespace InSoMniA{
		
		template<typename fc>
		class optimizer{
		private:
			std::shared_ptr<fc> _object;
			struct iter_info{
				size_t _max_iter;
				double _threshold;
				
				double _last_value;

				size_t _current_iter;
				Eigen::VectorXd _current_point;
				double _current_value;
				Eigen::VectorXd _current_direction;
			} _iter_info;

			std::vector<std::shared_ptr<func<foc>> > _eqc;

		public:
			optimizer(const std::shared_ptr<fc>& object);

			optimizer(const std::shared_ptr<fc>& object,
					 const std::vector<std::shared_ptr<func<foc>>>& eqc);

			void initialize(const Eigen::VectorXd& init_point);

			bool solve(Eigen::VectorXd& end_point, double& end_value,
					   const double threshold, const size_t max_iter);

			bool solve_eqc(Eigen::VectorXd& end_point, double& end_value);

			~optimizer();
		};

		template<typename fc>
		optimizer<fc>::optimizer(const std::shared_ptr<fc>& object):
			_object(object){
		
		}
		
		template<typename fc>
		optimizer<fc>::optimizer(const std::shared_ptr<fc>& object,
			const std::vector<std::shared_ptr<func<foc>> >& eqc):
			_object(object),
			_eqc(eqc){

		}

		template<typename fc>
		void optimizer<fc>::initialize(const Eigen::VectorXd& init_point){
			_iter_info._current_point = init_point;
			_iter_info._current_iter = 0;
			_iter_info._max_iter = 10000;
			_iter_info._threshold = 1e-10;
			_iter_info._last_value = std::numeric_limits<double>::max();
			_object->value(init_point, _iter_info._current_value);

			std::cerr << "init value: " << _iter_info._current_value << std::endl;
		}
		
		// this function implements steepest descent method
		template<>
		bool optimizer<func<foc> >::solve(Eigen::VectorXd& end_point, double& end_value,
							   const double threshold, const size_t max_iter){
			_iter_info._max_iter = max_iter;
			_iter_info._threshold = threshold;

			while( _iter_info._current_iter < _iter_info._max_iter 
				&& _iter_info._last_value - _iter_info._current_value > threshold){
				// step1: decide direction
				_object->gradient(_iter_info._current_point, _iter_info._current_direction);
				_iter_info._current_direction = -_iter_info._current_direction;

				std::cerr << "step: " << _iter_info._current_iter << std::endl;
				std::cerr << "direction: " << _iter_info._current_direction.transpose() << std::endl;

				// step2: decide step size
				double value;

				std::function<void(double, double&)> phi_func = [&object = _object, 
					 			 &direction = _iter_info._current_direction, 
					 			 &point = _iter_info._current_point] 
								 (double alpha, double& value){
								 	// std::cout << "eval point: " << (point + alpha*direction).transpose() << std::endl;
									object->value(point + alpha*direction, value);
									// std::cout << "eval: " << value << std::endl;
								};

				std::function<void(double, double&)> d_phi_func = [&object = _object,
					 			   &direction = _iter_info._current_direction,
								   &point = _iter_info._current_point]
								   (double alpha, double& derivative){
								   		Eigen::VectorXd gred;
								   		object->gradient(point + alpha*direction, gred);
								   		derivative = gred.dot(direction);
								   };		

				double step_length = line_search(value,
					phi_func,
					d_phi_func, 
					_iter_info._last_value);

				std::cerr << "step length: " << step_length << std::endl;

				// step3: update status
				_iter_info._current_iter++;
				_iter_info._last_value = _iter_info._current_value;
				_iter_info._current_value = value;
				_iter_info._current_point = 
					_iter_info._current_point + step_length * _iter_info._current_direction;

				std::cerr << "current point: " << _iter_info._current_point.transpose() << std::endl;
				// std::cerr << "last value: " << _iter_info._last_value << std::endl;
				std::cerr << "current value: " << _iter_info._current_value << std::endl;
			}

			end_point = _iter_info._current_point;
			end_value = _iter_info._current_value;
			if(_iter_info._last_value - _iter_info._current_value < threshold)
				return true;
			else
				return false;
		}
		
		template<>
		bool optimizer<func<soc> >::solve(Eigen::VectorXd& end_point, double& end_value,
							   const double threshold, const size_t max_iter){
			_iter_info._max_iter = max_iter;
			_iter_info._threshold = threshold;

			while( _iter_info._current_iter < _iter_info._max_iter 
				&& _iter_info._last_value - _iter_info._current_value > threshold){
				// step1: decide direction
				Eigen::MatrixXd hessian;
				Eigen::VectorXd gradient;

				_object->gradient(_iter_info._current_point, gradient);
				_object->hessian(_iter_info._current_point, hessian);

				_iter_info._current_direction = - hessian.inverse() * gradient;

				std::cerr << "step: " << _iter_info._current_iter << std::endl;
				std::cerr << "direction: " << _iter_info._current_direction.transpose() << std::endl;
				// std::cerr << "hessian: " << hessian << std::endl;
				// step2: decide step size
				double value;
				// Eigen::VectorXd gradient;

				std::function<void(double, double&)> phi_func = [&object = _object, 
					 			 &direction = _iter_info._current_direction, 
					 			 &point = _iter_info._current_point] 
								 (double alpha, double& value){
								 	// std::cout << "eval point: " << (point + alpha*direction).transpose() << std::endl;
									object->value(point + alpha*direction, value);
									// std::cout << "eval: " << value << std::endl;
								};

				std::function<void(double, double&)> d_phi_func = [&object = _object,
					 			   &direction = _iter_info._current_direction,
								   &point = _iter_info._current_point]
								   (double alpha, double& derivative){
								   		Eigen::VectorXd gred;
								   		object->gradient(point + alpha*direction, gred);
								   		derivative = gred.dot(direction);
								   };		

				double step_length = line_search(value,
					phi_func,
					d_phi_func, 
					std::numeric_limits<double>::max());

				std::cerr << "step length: " << step_length << std::endl;

				// step3: update status
				_iter_info._current_iter++;
				_iter_info._last_value = _iter_info._current_value;
				_iter_info._current_value = value;
				_iter_info._current_point = 
					_iter_info._current_point + step_length * _iter_info._current_direction;

				std::cerr << "current point: " << _iter_info._current_point.transpose() << std::endl;
				// std::cerr << "last value: " << _iter_info._last_value << std::endl;
				std::cerr << "current value: " << _iter_info._current_value << std::endl;
			}

			end_point = _iter_info._current_point;
			end_value = _iter_info._current_value;
			if(_iter_info._last_value - _iter_info._current_value < threshold)
				return true;
			else
				return false;

		}
		
		// this method only solve qp with linear equal constraints
		template<>
		bool optimizer<func<soc> >::solve_eqc(Eigen::VectorXd& end_point, double& end_value){

			Eigen::MatrixXd hessian;
			_object->hessian(_iter_info._current_point, hessian);

			// build kkt matrix
			Eigen::MatrixXd kkt_matrix;
			kkt_matrix.resize(hessian.rows() + _eqc.size(), hessian.cols() + _eqc.size());
			kkt_matrix.setZero();

			kkt_matrix.block(0, 0, hessian.rows(), hessian.cols()) = hessian;
			Eigen::MatrixXd A;
			build_A_from_eqc(_eqc, A);
			
			kkt_matrix.block(hessian.rows(), 0, _eqc.size(), hessian.cols()) = A;
			kkt_matrix.block(0, hessian.cols(), hessian.rows(), _eqc.size()) = A.transpose();

			// build g and h
			Eigen::VectorXd g, h;
			_object->gradient(_iter_info._current_point, g);
			build_h_from_eqc(_eqc, _iter_info._current_point, h);
			Eigen::VectorXd gh;
			gh.resize(g.rows() + h.rows());
			gh.segment(0, g.rows()) = g;
			gh.segment(g.rows(), h.rows()) = h;

			// solve kkt matrix
			Eigen::VectorXd x;
			x = kkt_matrix.inverse() * gh;

			// check result
			Eigen::VectorXd p, lambda;
			p.resize(g.rows());
			lambda.resize(h.rows());
			
			p = -x.segment(0, g.rows());
			lambda = x.segment(g.rows(), h.rows());

			_iter_info._current_point += p;
			_object->value(_iter_info._current_point, _iter_info._current_value);
			end_point = _iter_info._current_point;
			end_value = _iter_info._current_value;
			return true;
		}

		
		template<typename fc>
		optimizer<fc>::~optimizer() { }
	}
}

#endif