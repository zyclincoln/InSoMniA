#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <set>
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
			std::vector<std::shared_ptr<func<foc>> > _ieqc;

		public:
			optimizer(const std::shared_ptr<fc>& object);

			optimizer(const std::shared_ptr<fc>& object,
					 const std::vector<std::shared_ptr<func<foc>>>& eqc);

			optimizer(const std::shared_ptr<fc>& object,
					const std::vector<std::shared_ptr<func<foc>>>& eqc,
					const std::vector<std::shared_ptr<func<foc>>>& ieqc);

			void initialize(const Eigen::VectorXd& init_point);

			bool solve(Eigen::VectorXd& end_point, double& end_value,
					   const double threshold, const size_t max_iter);

			bool solve_eqc(Eigen::VectorXd& end_point, double& end_value);

			bool solve_active_set(Eigen::VectorXd& end_point, double& end_value,
						const double threshold, const size_t max_iter);

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
		optimizer<fc>::optimizer(const std::shared_ptr<fc>& object,
			const std::vector<std::shared_ptr<func<foc>> >& eqc,
			const std::vector<std::shared_ptr<func<foc>> >& ieqc):
			_object(object),
			_eqc(eqc),
			_ieqc(ieqc){
		}

		template<typename fc>
		void optimizer<fc>::initialize(const Eigen::VectorXd& init_point){
			_iter_info._current_point = init_point;
			_iter_info._current_iter = 0;
			_iter_info._max_iter = 10000;
			_iter_info._threshold = 1e-10;
			_iter_info._last_value = std::numeric_limits<double>::max();
			_object->value(init_point, _iter_info._current_value);
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

		template<>
		bool optimizer<func<soc> >::solve_active_set(Eigen::VectorXd& end_point, double& end_value,
													const double threshold, const size_t max_iter){
			_iter_info._max_iter = max_iter;
			_iter_info._threshold = threshold;

			std::set<size_t> working_set_index;

			while(_iter_info._current_iter < _iter_info._max_iter 
				&& _iter_info._last_value - _iter_info._current_value > threshold){
				// build kkt system from working set and rhs

				{
					std::cerr << "iter: " << _iter_info._current_iter << std::endl;
					std::cerr << "working_set: ";
					for(auto iter = working_set_index.cbegin(); iter != working_set_index.cend(); iter++){
						std::cerr << *iter << ", ";
					}
					std::cerr << std::endl;
					std::cerr << "current_point: " << _iter_info._current_point.transpose() << std::endl;
					std::cerr << "current_value: " << _iter_info._current_value << std::endl;
				}

				Eigen::MatrixXd hessian;
				_object->hessian(_iter_info._current_point, hessian);

				Eigen::MatrixXd kkt_matrix;
				kkt_matrix.resize(hessian.rows() + _eqc.size() + working_set_index.size(),
								  hessian.cols() + _eqc.size() + working_set_index.size());
				kkt_matrix.setZero();
				kkt_matrix.block(0, 0, hessian.rows(), hessian.cols()) = hessian;
				// std::cout << "kkt matrix: " << std::endl << kkt_matrix << std::endl;
				
				if(_eqc.size() > 0){
					Eigen::MatrixXd A1;
					build_A_from_eqc(_eqc, A1);
					kkt_matrix.block(hessian.rows(), 0, _eqc.size(), hessian.cols()) = A1;
					kkt_matrix.block(0, hessian.cols(), hessian.rows(), _eqc.size()) = A1.transpose();
				}
				
				if(working_set_index.size() > 0){
					Eigen::MatrixXd A2;
					build_A_from_workingset(_ieqc, working_set_index, A2);
					kkt_matrix.block(hessian.rows() + _eqc.size(), 0, working_set_index.size(), hessian.cols()) = A2;
					kkt_matrix.block(0, hessian.cols() + _eqc.size(), hessian.rows(), working_set_index.size()) = A2.transpose();
				}
				// std::cout << "kkt matrix: " << std::endl << kkt_matrix << std::endl;

				Eigen::VectorXd g;
				_object->gradient(_iter_info._current_point, g);
				Eigen::VectorXd h1;
				if(_eqc.size() > 0){
					build_h_from_eqc(_eqc, _iter_info._current_point, h1);
				}
				Eigen::VectorXd h2;
				if(working_set_index.size() > 0){
					build_h_from_workingset(_ieqc, working_set_index, _iter_info._current_point, h2);
				}

				Eigen::VectorXd gh;
				gh.resize(g.rows() + _eqc.size() + working_set_index.size());
				gh.segment(0, g.rows()) = g;
				// std::cout << "gh: " << gh.transpose() << std::endl;
				gh.segment(g.rows(), _eqc.size()) = h1;
				if(working_set_index.size() > 0){
					gh.segment(g.rows() + _eqc.size(), working_set_index.size()) = h2;
				}
				// std::cout << "kkt matrix: " << std::endl << kkt_matrix << std::endl;
				Eigen::VectorXd x; 
				x = kkt_matrix.inverse() *gh;
				// std::cout << "gh: " << gh.transpose() << std::endl;
				Eigen::VectorXd p, lambda;
				p.resize(g.rows());
				lambda.resize(x.rows() - g.rows());
				p = -x.segment(0, g.rows());
				lambda = x.segment(g.rows(), x.rows() - g.rows());
				// std::cout << "p: " << p.transpose() << std::endl;

				// check the length of k
				if(p.norm() < 1e-10){
				// if k is zero

					// check if kkt parameter is satisfied
					// if yes, return
					// if no, remove one from working set
					auto min_iter = working_set_index.end();
					double min_lambda = 0;
					int i = 0;
					for(auto iter = working_set_index.begin(); iter != working_set_index.end(); iter++, i++){
						if(lambda(_eqc.size() + i, 0) < min_lambda){
							min_lambda = lambda(_eqc.size() + i, 0);
							min_iter = iter;
						}
					}

					if(min_iter != working_set_index.end()){
						working_set_index.erase(min_iter);
					}
					else{
						break;
					}
				}
				else{
				// else
					// calculate alpha from p

					// if meet constraints
					// add constraints into working set
					double alpha = 1;
					auto constraint_iter = _ieqc.end();
					int i = 0;
					for(auto iter = _ieqc.begin(); iter != _ieqc.end(); iter++, i++){
						if(working_set_index.find(i) != working_set_index.end())
							continue;
						Eigen::VectorXd gradient;
						(*iter)->gradient(_iter_info._current_point, gradient);
						// std::cout << "gradient: " << gradient.transpose() << std::endl;
						// std::cout << "gradient value decrease: " << gradient.transpose() * p << std::endl;
						if(gradient.transpose() * p >= 0)
							continue;

						double value;
						(*iter)->value(_iter_info._current_point, value);

						double c_alpha = -value / (gradient.transpose() * p);
						
						// std::cerr << "gradient: " << gradient.transpose() << std::endl;
						// std::cerr << "value: " << value << std::endl;
						// std::cerr << "c_alpha: " << c_alpha << std::endl;
						assert(c_alpha > 0);
						if(c_alpha < alpha){
							alpha = c_alpha;
							constraint_iter = iter;
						}
					}

					_iter_info._last_value = _iter_info._current_value;
					_iter_info._current_point += alpha*p;

					_object->value(_iter_info._current_point, _iter_info._current_value);

					if(alpha != 1){
						working_set_index.insert(constraint_iter - _ieqc.begin());
					}
				}

				{
					std::cerr << "working_set: ";
					for(auto iter = working_set_index.cbegin(); iter != working_set_index.cend(); iter++){
						std::cerr << *iter << ", ";
					}
					std::cerr << std::endl;
					std::cerr << "current_point: " << _iter_info._current_point.transpose() << std::endl;
					std::cerr << "current_value: " << _iter_info._current_value << std::endl;

				}
				_iter_info._current_iter++;
			}

			end_point = _iter_info._current_point;
			end_value = _iter_info._current_value;
			return true;
		}
		
		template<typename fc>
		optimizer<fc>::~optimizer() { }
	}
}

#endif