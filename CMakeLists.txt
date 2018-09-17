cmake_minimum_required(VERSION 3.5)

project(InSoMniA)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

FIND_PACKAGE(PETSc 3.7 REQUIRED)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
ADD_DEFINITIONS(${PETSC_DEFINITIONS})

include_directories(include)
file(GLOB INSOMNIA_HEADER 
	"include/*/*.h"
)
enable_testing()

add_subdirectory(src)
add_subdirectory(test)