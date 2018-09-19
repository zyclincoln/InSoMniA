#include "utility/matrix_utility.h"

namespace zyclincoln{
	namespace InSoMniA{
		bool merge_vector(const std::vector<Eigen::VectorXd>& candidates,
						  Eigen::VectorXd& result){
			size_t size = 0;
			for(auto i = candidates.cbegin(); i != candidates.cend(); i++){
				size += i->rows();
			}

			result.resize(size);

			size_t cursor = 0;
			for(auto i = candidates.cbegin(); i!= candidates.cend(); i++){
				result.segment(cursor, i->rows()) = *i;
				cursor += i->rows();
			}

			return true;
		}
	}
}