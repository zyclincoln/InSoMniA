#include "svm/svm.hpp"

#include <iostream>
#include <cassert>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace zyclincoln{
	namespace InSoMniA{
		double svr::evaluate(const Eigen::VectorXd& x) const{
			double result = 0;
			for(auto i = _F_set.begin(); i != _F_set.end(); ++i)
				result += _a(*i) * kernel_eval(_x.col(*i), x);
			for(auto i = _Ac_set.begin(); i != _Ac_set.end(); ++i)
				result += _a(*i) * kernel_eval(_x.col(*i), x);
			return result;
		}

		double svr::kernel_eval(const Eigen::VectorXd& xi, const Eigen::VectorXd& xj) const{
			if(_kernel == KernelType::linear)
				return xi.dot(xj);
		}

		double svr::predict(const Eigen::VectorXd& x) const{
			return evaluate(x);
		}

		void svr::train(){
			// build init set
			std::set<size_t> A0_set, pAc_set, nAc_set;
			{
				set<size_t> init_nAc_set, init_pAc_set, init_A0_set, init_F_set;
				for(int i = 0; i < _x.cols(); ++i){
					double e = evaluate(_x.col(i));
					double error = e - _y(i);
					if(-_epsilon - error > 0){
						init_pAc_set.insert(i);
						cout << "[ SVM train ] insert item " << i << " into positive Ac set" << endl;
					}
					else if(-_epsilon + error > 0){
						init_nAc_set.insert(i);
						cout << "[ SVM train ] insert item " << i << " into negative Ac set" << endl;
					}
					else{
						init_F_set.insert(i);
						cout << "[ SVM train ] insert item " << i << " into F set" << endl;
					}
				}
				nAc_set.insert(init_nAc_set.begin(), init_nAc_set.end());
				pAc_set.insert(init_pAc_set.begin(), init_pAc_set.end());
				_F_set.insert(init_F_set.begin(), init_F_set.end());
				_Ac_set.insert(pAc_set.begin(), pAc_set.end());
				_Ac_set.insert(nAc_set.begin(), nAc_set.end());
			}

			size_t iterate_num = 0;
			// begin interate
			while(iterate_num < _max_iterate_num){
				cout << "================================================" << endl;
				++iterate_num;
				cout << "[ SVM train ] iter: " << iterate_num << endl;
				// build kkt matrix
				Eigen::VectorXd new_a = _a;
				{
					size_t free_set_size = _F_set.size();
					Eigen::MatrixXd H;
					Eigen::VectorXd b;
					H.resize(free_set_size, free_set_size);
					b.resize(free_set_size);

					int index_i = 0;
					for(auto i = _F_set.begin(); i != _F_set.end(); ++i, ++index_i){
						int index_j = 0;
						for(auto j = _F_set.begin(); j != _F_set.end(); ++j, ++index_j)
							H(index_i, index_j) = 
								kernel_eval(_x.col(*i), _x.col(*j));

						b(index_i) = _y(*i);
						for(auto j = _Ac_set.begin(); j != _Ac_set.end(); ++j)
							b(index_i) -= _a(*j)*kernel_eval(_x.col(*i), _x.col(*j));
						if(_a(*i) >= 0)
							b(index_i) -= _epsilon;
						else
							b(index_i) += _epsilon;
					}
					Eigen::VectorXd a;
					a.resize(free_set_size);
					a = H.llt().solve(b);
					// cout << "H: " << endl;
					// cout << H << endl;
					// cout << "b: " << endl;
					// cout << b << endl;
					// cout << "a: " << endl;
					// cout << a << endl;
					double relative_error = (H*a - b).norm()/b.norm();
					cout << "[ SVM train ] solving relative error: " << relative_error << endl;
					// apply back to global a
					int index = 0;
					for(auto i = _F_set.begin(); i != _F_set.end(); ++i, ++index){
						new_a(*i) = a(index);
					}
				}
				// check within feasible region
				{
					double miu = 1;
					int miu_index = -1;
					// type1: to pAc, type2: to nAc
					int move_type = -1;

					set<size_t> temp_A0_set;
					for(auto i = _F_set.begin(); i != _F_set.end(); ++i){
						if(fabs(new_a(*i)) < 1e-3){
						// if(new_a(*i) == 0){
							temp_A0_set.insert(*i);
							continue;
						}

						int current_move_type = -1;
						double new_miu = 1;
						if(_F_set.find(*i) != _F_set.end()){
							if(new_a(*i) < _C && new_a(*i) > -_C)
								continue;
							// move to pAc
							if(new_a(*i) >= _C){
								new_miu = (_C-_a(*i))/(new_a(*i)-_a(*i));
								assert(new_miu > 0);
								assert(new_miu < 1);
								current_move_type = 1;
							}
							else if(new_a(*i) <= -_C){
								new_miu = (-_C - _a(*i))/(new_a(*i)-_a(*i));
								assert(new_miu > 0);
								assert(new_miu < 1);
								current_move_type = 2;
							}
						}
						else{
							cout << "[ SVM train ] ERROR: point not in F set: " << *i << endl;
							assert(false);
						}

						if(new_miu < miu){
							miu = new_miu;
							miu_index = *i;
							move_type = current_move_type;
						}
					}

					for(auto i = temp_A0_set.begin(); i != temp_A0_set.end(); ++i){
						A0_set.insert(*i);
						_F_set.erase(*i);
						_a(*i) = 0;
					}

					if(miu == 1){
						cout << "[ SVM train ] with in feasible region" << endl;
						_a = new_a;
						// cout << "A: " << _a << endl;
						// goto kkt condition check step
					}
					else{
						// check miu_index and move it from free set to Ac set
						assert(_F_set.find(miu_index) != _F_set.end());
						cout << "[ SVM train ] not in feasible region" << endl; 
						_F_set.erase(miu_index);
						switch(move_type){
							case 1: pAc_set.insert(miu_index);
									_Ac_set.insert(miu_index);
									cout << "[ SVM train ] move " << miu_index << " to pAc set" << endl;
									break;
							case 2: nAc_set.insert(miu_index);
									_Ac_set.insert(miu_index);
									cout << "[ SVM train ] move " << miu_index << " to nAc set" << endl;
									break;
							default:cout << "[ SVM train ] unknown move type! " << move_type << endl;
									assert(false);
						}

						_a = miu * new_a + (1-miu) * _a;
						if(move_type == 0)
							_a(miu_index) = 0;

						// cout << "A: " << _a << endl;
						// continue to next step, bypass kkt condition check
						continue;
					}
				}
				// check kkt condition
				{
					int most_negative_index = -1;
					double most_negative_value = 0;
					double most_negative_error = 0;
					for(auto i = A0_set.begin(); i != A0_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double lambda = error < 0 ? _epsilon + error : _epsilon - error;
						if(lambda < most_negative_value){
							most_negative_value = lambda;
							most_negative_index = *i;
						}
					}

					for(auto i = pAc_set.begin(); i != pAc_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double miu = error < 0 ? -_epsilon + error : -_epsilon - error;
						if(miu < most_negative_value){
							most_negative_value = miu;
							most_negative_index = *i;
						}
					}

					for(auto i = nAc_set.begin(); i != nAc_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double miu = error < 0 ? -_epsilon + error : -_epsilon - error;
						if(miu < most_negative_value){
							most_negative_value = miu;
							most_negative_index = *i;
						}
					}

					if(most_negative_index >= 0){
						A0_set.erase(most_negative_index);
						nAc_set.erase(most_negative_index);
						pAc_set.erase(most_negative_index);
						_Ac_set.erase(most_negative_index);
						_F_set.insert(most_negative_index);
						cout << "[ SVM train ] violate kkt condition, value: " << most_negative_value << ", index: " << most_negative_index << endl;
					}
					else{
						// not violate kkt condition, find optimum
						cout << "[ SVM train ] get optimum, finish" << endl;
						cout << "[ SVM train ] A: " << _a << endl;
						cout << "[ SVM train ] F set size: " << _F_set.size() << endl;
						cout << "[ SVM train ] Ac set size: " << _Ac_set.size() << endl;
						cout << "[ SVM train ] pAc set size: " << pAc_set.size() << endl;
						cout << "[ SVM train ] nAc set size: " << nAc_set.size() << endl;
						cout << "[ SVM train ] A0 set size: " << A0_set.size() << endl;
						return;
					}
				}
			}
			return;
		}
	}
}