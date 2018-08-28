cmake_minimum_required(VERSION 3.5)

project(InSoMniA)

set (CMAKE_CXX_STANDARD 11)

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

include_directories(include)
file(GLOB INSOMNIA_HEADER 
	"include/*/*.h"
)

add_subdirectory(src)
add_subdirectory(test)