project(GuidedFilter)

file(GLOB sources *.cc)
add_executable(${PROJECT_NAME} ${sources})

add_subdirectory(gfTest)
add_subdirectory(gfTime)
add_subdirectory(gfWeights)
