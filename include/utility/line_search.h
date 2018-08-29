#ifndef _LINE_SEARCH_H_
#define _LINE_SEARCH_H_

#include <cmath>
#include <limits>
#include <iostream>
#include <functional>
#include "Eigen/Core"
#include "interpolation.h"

namespace zyclincoln{
	namespace InSoMniA{

		double zoom(double& phi_alpha, double alpha_lo, double alpha_hi,
					std::function< void(double, double&) >& value,
					std::function< void(double, double&) >& derivative, 
					double phi_alpha_lo, double phi_alpha_hi, const double phi_alpha0,
					double d_phi_alpha_lo, double d_phi_alpha_hi, const double d_phi_alpha0){
			double alpha, d_phi_alpha;
			double c1 = 1e-4, c2 = 0.9;

			while(true){
				// std::cerr << alpha_lo << " " << alpha_hi << " " << phi_alpha_lo << " "
				// 		  << phi_alpha_hi << " " << d_phi_alpha_lo << " " << d_phi_alpha_hi << std::endl;
				// std::cerr << phi_alpha0 << " " << d_phi_alpha0 << std::endl;

				alpha = cubic_interpolation(alpha_lo, alpha_hi, phi_alpha_lo, phi_alpha_hi,
				  						    d_phi_alpha_lo, d_phi_alpha_hi);
				std::cerr << "zoom alpha: " << alpha << std::endl;
				value(alpha, phi_alpha);
				derivative(alpha, d_phi_alpha);
				std::cerr << "d_phi_alpha: " << d_phi_alpha << std::endl;
				std::cerr << "ref: " << -c2*d_phi_alpha0 << std::endl;
				// std::cout << "eval: phi_alpha: " << phi_alpha << std::endl;
				// std::cout << "eval: phi_alpha0: " << phi_alpha0 << std::endl; 
				// std::cout << "eval: ref: " << phi_alpha0 + c1 * alpha * d_phi_alpha0 << std::endl;

				if(phi_alpha > phi_alpha0 + c1 * alpha * d_phi_alpha0 || phi_alpha >= phi_alpha_hi){
					d_phi_alpha_hi = d_phi_alpha;
					phi_alpha_hi = phi_alpha;
					alpha_hi = alpha;
				}
				else{
					if(fabs(d_phi_alpha) <= -c2*d_phi_alpha0)
						return alpha;
					if(d_phi_alpha * (alpha_hi - alpha_lo) >= 0)
						alpha_hi = alpha_lo;
					alpha_lo = alpha;
				}

				if(fabs(alpha_hi - alpha_lo) < 1e-10)
					return alpha;
			}
		}


		double init_step_length(double v0, double v1, double d1){
			return std::min(1.0, 1.01 * quadratic_interpolation(v0, v1, d1));
		}

		double init_step_length(){
			return 1;
		}

		double line_search(double &phi_alpha,
							std::function< void(double, double&) >& value,
							std::function< void(double, double&) >& derivative,
							const double last_value = std::numeric_limits<double>::max()
							){
			double alpha0=0, alpha, alpha1=0, alpha_max=100;
			double phi_alpha0, phi_alpha1;
			double d_phi_alpha, d_phi_alpha0, d_phi_alpha1;
			int iter = 0;
			double c1 = 1e-4, c2 = 0.9;

			value(0, phi_alpha0);
			derivative(0, d_phi_alpha0);
			d_phi_alpha1 = d_phi_alpha0;
			phi_alpha1 = phi_alpha0;

			if(last_value == std::numeric_limits<double>::max()){
				alpha = init_step_length();
			}
			else{
				alpha = init_step_length(last_value, phi_alpha0, d_phi_alpha0);
			}

			std::cerr << "init alpha: " << alpha << std::endl;
			while(true){
				// std::cerr << "alpha: " << alpha << std::endl;
				value(alpha, phi_alpha);
				derivative(alpha, d_phi_alpha);
				// std::cout << "alpha1: " << alpha1 << std::endl;
				// std::cout << "phi alpha: " << phi_alpha << std::endl;
				// std::cout << "phi alpha1: " << phi_alpha1 << std::endl;
				// std::cout << "d phi alpha: " << d_phi_alpha << std::endl;
				// std::cout << "d phi alpha1: " << d_phi_alpha1 << std::endl;
				std::cout << "d phi alpha0: " << d_phi_alpha0 << std::endl;
				if(phi_alpha > phi_alpha0 + c1*alpha*d_phi_alpha0 || 
					(iter > 0 && phi_alpha > phi_alpha1))
					return zoom(phi_alpha, alpha1, alpha, value, derivative,
								phi_alpha1, phi_alpha, phi_alpha0,
								d_phi_alpha1, d_phi_alpha, d_phi_alpha0);

				if(fabs(d_phi_alpha) <= -c2*d_phi_alpha0)
					return alpha;

				if(d_phi_alpha >= 0)
					return zoom(phi_alpha, alpha, alpha1, value, derivative,
								phi_alpha, phi_alpha1, phi_alpha0,
								d_phi_alpha, d_phi_alpha1, d_phi_alpha0);

				alpha1 = alpha;
				alpha = std::min(alpha_max, 2*alpha);

				iter++;
			}
		}

	}
}

#endif