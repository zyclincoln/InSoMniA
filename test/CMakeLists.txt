add_executable(OptimizerTest OptimizerTest.cpp ${INSOMNIA_HEADER})
add_executable(Eqc_KKT_Test Eqc_KKT_Test.cpp ${INSOMNIA_HEADER})

target_link_libraries(OptimizerTest InSoMniA)
target_link_libraries(Eqc_KKT_Test InSoMniA)