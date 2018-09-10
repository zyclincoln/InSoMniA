#ifndef _INTERPOLATION_H_
#define _INTERPOLATION_H_

#include <cassert>
#include <cmath>
#include <iostream>
/*
 *  p means the point
 *  v means the function value
 *  d means the differential of function
 */


namespace zyclincoln{
	namespace InSoMniA{

		double cubic_interpolation(double p0, double p1, 
								   double v0, double v1, 
								   double d0, double d1){
			
			double a1 = d0 + d1 - 3*(v0-v1)/(p0-p1);
			double a2 = sqrt(a1*a1 - d0*d1);
			// std::cerr << "a1: " << a1 << " " << "a2: " << a2 << std::endl;
			double value = p1 - (p1 - p0)*(d1 + a2 - a1)/(d1 - d0 + 2*a2);

			// if numerical problem happened
			// assert(fabs(value) < 1e5);
			// std::cerr << "a i+1 : " << value << std::endl;
			return value;
		}

		double quadratic_interpolation(double v0, double v1, double d1){
			double value = (v1 - v0)/d1;
			// if d1 is too little
			// assert(fabs(value) < 1e5);

			return value;
		}

	}
}

#endif 