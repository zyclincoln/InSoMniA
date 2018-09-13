add_executable(OptimizerTest OptimizerTest.cpp ${INSOMNIA_HEADER})
add_executable(Eqc_KKT_Test Eqc_KKT_Test.cpp ${INSOMNIA_HEADER})
add_executable(petsc_test petsc_test.cpp)
add_executable(petsc_test_parallel petsc_test_parallel.cpp)

target_link_libraries(OptimizerTest InSoMniA)
target_link_libraries(Eqc_KKT_Test InSoMniA)
target_link_libraries(petsc_test ${PETSC_LIBRARIES})
target_link_libraries(petsc_test_parallel ${PETSC_LIBRARIES})