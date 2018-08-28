#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <memory>
#include "Eigen/Core"
#include "function/function.h"

namespace zyclincoln{

	namespace InSoMniA{
	
		template<typename fc>
		class optimizer{
		private:
			std::shared_ptr<fc> _object;
			struct iter_info{
				size_t _current_iter;
				size_t _max_iter;
				Eigen::VectorXd _current_point;
				double _current_value;
				double _current_derivative;
			} _iter_info;

		public:
			optimizer(const std::shared_ptr<fc>& object);

			void init_point(const Eigen::VectorXd& init_point);

			bool solve(Eigen::VectorXd& end_point, double& end_value,
					   const double threshold, const size_t max_iter);

			~optimizer();
		};

		template<typename fc>
		optimizer<fc>::optimizer(const std::shared_ptr<fc>& object):
			_object(object){
		
		}
		
		template<typename fc>
		void optimizer<fc>::init_point(const Eigen::VectorXd& init_point){
			_iter_info._current_point = init_point;
		}
		
		template<>
		bool optimizer<func<foc> >::solve(Eigen::VectorXd& end_point, double& end_value,
							   const double threshold, const size_t max_iter){
			return false;
		}
		
		template<>
		bool optimizer<func<soc> >::solve(Eigen::VectorXd& end_point, double& end_value,
							   const double threshold, const size_t max_iter){
			return false;
		}
		
		template<typename fc>
		optimizer<fc>::~optimizer() { }
	}
}

#endif