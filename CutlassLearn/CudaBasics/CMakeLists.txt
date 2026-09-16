add_executable(VectorAdd VectorAdd.cu)
add_executable(Reduction Reduction.cu)

foreach(target VectorAdd Reduction)
  target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/Common)
endforeach()
