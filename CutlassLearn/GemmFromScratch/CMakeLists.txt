add_executable(Gemm Gemm.cu)
target_include_directories(Gemm PRIVATE ${CMAKE_SOURCE_DIR}/Common)
target_link_libraries(Gemm PRIVATE cublas)
