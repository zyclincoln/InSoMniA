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
			std::set<size_t> pF_set, nF_set, A0_set, pAc_set, nAc_set;
			{
				set<size_t> init_nAc_set, init_pAc_set, init_A0_set, init_pF_set, init_nF_set;
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
						if(error < 0){
							init_pF_set.insert(i);
							cout << "[ SVM train ] insert item " << i << " into positive F set" << endl;
						}
						else{
							init_nF_set.insert(i);
							cout << "[ SVM train ] insert item " << i << " into negative F set" << endl;
						}
					}
				}
				nAc_set.insert(init_nAc_set.begin(), init_nAc_set.end());
				pAc_set.insert(init_pAc_set.begin(), init_pAc_set.end());
				pF_set.insert(init_pF_set.begin(), init_pF_set.end());
				nF_set.insert(init_nF_set.begin(), init_nF_set.end());
				_F_set.insert(pF_set.begin(), pF_set.end());
				_F_set.insert(nF_set.begin(), nF_set.end());
				_Ac_set.insert(pAc_set.begin(), pAc_set.end());
				_Ac_set.insert(nAc_set.begin(), nAc_set.end());
				for(auto i = pAc_set.begin(); i != pAc_set.end(); ++i){
					_a[*i] = _C;
				}
				for(auto i = nAc_set.begin(); i != nAc_set.end(); ++i){
					_a[*i] = -_C;
				}
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
						if(pF_set.find(*i) != pF_set.end()) // in positive f set
							b(index_i) -= _epsilon;
						else
							b(index_i) += _epsilon;
					}

					Eigen::VectorXd a;
					a.resize(free_set_size);
					a = H.llt().solve(b);
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
					// type 0: to A0, type1: to pAc, type2: to nAc
					int move_type = -1;
					for(auto i = _F_set.begin(); i != _F_set.end(); ++i){
						int current_move_type = -1;
						double new_miu = 1;
						std::cout << "i:" << *i << endl;
						std::cout << "new_a(i): " << new_a(*i) << endl;
						std::cout << "a(i): " << _a(*i) << endl;
						if(pF_set.find(*i) != pF_set.end()){
							if(new_a(*i) < _C && new_a(*i) > 0)
								continue;
							// move to pAc
							if(new_a(*i) >= _C){
								new_miu = (_C-_a(*i))/(new_a(*i)-_a(*i));
								assert(new_miu > 0);
								assert(new_miu < 1);
								current_move_type = 1;
							}
							// move to A0
							else{
								new_miu = -_a(*i)/(new_a(*i) - _a(*i));
								assert(new_miu > 0);
								assert(new_miu < 1);	
								current_move_type = 0;
							}
						}
						else if(nF_set.find(*i) != nF_set.end()){
							if(new_a(*i) < 0 && new_a(*i) > -_C)
								continue;
							// move to nAc
							if(new_a(*i) <= -_C){
								new_miu = (-_C - _a(*i))/(new_a(*i)-_a(*i));
								assert(new_miu > 0);
								assert(new_miu < 1);
								current_move_type = 2;
							}
							// move to A0
							else{
								new_miu = -_a(*i)/(new_a(*i) - _a(*i));
								std::cout << _a(*i) << std::endl;
								std::cout << new_a(*i) << std::endl;
								assert(new_miu > 0);
								assert(new_miu < 1);
								current_move_type = 0;
							}
						}
						else{
							cout << "[ SVM train] ERROR: point in F set but not in nF or pF: " << *i << endl;
							assert(false);
						}

						if(new_miu < miu){
							miu = new_miu;
							miu_index = *i;
							move_type = current_move_type;
						}
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
						pF_set.erase(miu_index);
						nF_set.erase(miu_index);
						switch(move_type){
							case 0: A0_set.insert(miu_index);
									cout << "[ SVM train ] move " << miu_index << " to A0 set" << endl;
									break;
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
					int move_type = -1; // 0 for pF, 1 for nF
					double most_negative_error = 0;
					for(auto i = A0_set.begin(); i != A0_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double lambda = error < 0 ? _epsilon + error : _epsilon - error;
						if(lambda < most_negative_value){
							most_negative_value = lambda;
							most_negative_index = *i;
							if(error > 0)
								move_type = 1;
							else
								move_type = 0;
						}
					}

					for(auto i = pAc_set.begin(); i != pAc_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double miu = error < 0 ? -_epsilon + error : -_epsilon - error;
						if(miu < most_negative_value){
							most_negative_value = miu;
							most_negative_index = *i;
							if(error > 0)
								move_type = 0;
							else
								move_type = 1;
						}
					}

					for(auto i = nAc_set.begin(); i != nAc_set.end(); ++i){
						double error = evaluate(_x.col(*i)) - _y(*i);
						double miu = error < 0 ? -_epsilon + error : -_epsilon - error;
						if(miu < most_negative_value){
							most_negative_value = miu;
							most_negative_index = *i;
							if(error > 0)
								move_type = 0;
							else
								move_type = 1;
						}
					}

					if(most_negative_index >= 0){
						A0_set.erase(most_negative_index);
						nAc_set.erase(most_negative_index);
						pAc_set.erase(most_negative_index);
						_Ac_set.erase(most_negative_index);
						_F_set.insert(most_negative_index);
						cout << "[ SVM train ] violate kkt condition, value " << most_negative_value << endl;
						switch(move_type){
							case 0: pF_set.insert(most_negative_index);
									// _a(most_negative_index) = _C;
									cout << "[ SVM train ] move " << most_negative_index << " into pF set" << endl;
									break;
							case 1:	nF_set.insert(most_negative_index);
									// _a(most_negative_index) = -_C;
									cout << "[ SVM train ] move " << most_negative_index << " into nF set" << endl;
									break;
							default:
									cout << "[ SVM train ] ERROR: unknown moving type: " << move_type << endl;
						}
					}
					else{
						// not violate kkt condition, find optimum
						cout << "[ SVM train ] get optimum, finish" << endl;
						cout << "[ SVM train ] F set size: " << _F_set.size() << endl;
						cout << "[ SVM train ] pF set size: " << pF_set.size() << endl;
						cout << "[ SVM train ] nF set size: " << nF_set.size() << endl;
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