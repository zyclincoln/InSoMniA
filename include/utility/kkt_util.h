#ifndef _KKT_UTIL_H_
#define _KKT_UTIL_H_

#include "Eigen/Core"
#include "Eigen/Dense"
#include <memory>
#include <vector>
#include <set>
#include "func/func.h"
#include <iostream>

namespace zyclincoln{
	namespace InSoMniA{

		bool build_A_from_eqc(std::vector<std::shared_ptr<func<foc>>>& eqc, Eigen::MatrixXd& A){
			if(eqc.size() == 0){
				A.resize(0, 0);
				return false;
			}

			Eigen::VectorXd gradient;
			Eigen::VectorXd point;
			size_t dimension = eqc[0]->dimension();

			point.resize(dimension);
			point.setZero();

			A.resize(eqc.size(), dimension);
			for(size_t i = 0; i < eqc.size(); i++){
				eqc[i]->gradient(point, gradient);
				A.block(i, 0, 1, dimension) = gradient.transpose();
			}
			return true;
		}

		bool build_h_from_eqc(std::vector<std::shared_ptr<func<foc>>>& eqc, Eigen::VectorXd& point, Eigen::VectorXd& h){
			if(eqc.size() == 0){
				h.resize(0);
				return false;
			}

			h.resize(eqc.size());

			for(int i =0; i < eqc.size(); i++){
				eqc[i]->value(point, h(i, 0));
			}
			return true;
		}

		bool build_A_from_workingset(std::vector<std::shared_ptr<func<foc>>>& ieqc, std::set<size_t>& working_set, Eigen::MatrixXd& A){
			if(working_set.size() == 0){
				A.resize(0, 0);
				return false;
			}

			Eigen::VectorXd point;
			size_t dimension = ieqc[0]->dimension();
			point.resize(dimension);
			point.setZero();

			Eigen::VectorXd gradient;
			A.resize(working_set.size(), dimension);
			int i = 0;
			for(auto iter = working_set.begin(); 
				iter != working_set.end(); 
				iter++, i++){

				ieqc[*iter]->gradient(point, gradient);
				A.block(i, 0, 1, dimension) = gradient.transpose();
			}
			return true;
		}

		bool build_h_from_workingset(std::vector<std::shared_ptr<func<foc>>>& ieqc, std::set<size_t>& working_set, Eigen::VectorXd& point, Eigen::VectorXd& h){
			if(working_set.size() == 0){
				h.resize(0);
				return false;
			}

			h.resize(working_set.size());
			int i = 0;
			for(auto iter = working_set.begin(); 
				iter != working_set.end(); 
				iter++, i++ ){

				ieqc[*iter]->value(point, h(i, 0));
			}
			return true;
		}
	}
}

#endif