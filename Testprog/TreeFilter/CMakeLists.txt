project(TreeFilter)

file(GLOB sources *.cc)
add_executable(${PROJECT_NAME} ${sources})

add_subdirectory(tfTest)
add_subdirectory(tfTime)
add_subdirectory(tfWeights)
