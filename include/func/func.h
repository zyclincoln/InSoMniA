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
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient) = 0;
			virtual size_t dimension() = 0;
			typedef foc FUNCTION_TYPE;
		};

		template<>
		class func<soc>{
		public:
			virtual bool value(const Eigen::VectorXd& point, double& value) = 0;
			virtual bool gradient(const Eigen::VectorXd& point, Eigen::VectorXd& gradient) = 0;
			virtual bool hessian(const Eigen::VectorXd& point, Eigen::MatrixXd& hessian) = 0;
			virtual size_t dimension() = 0;
			typedef soc FUNCTION_TYPE;
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