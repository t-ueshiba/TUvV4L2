project(stereoV4L2)

file(GLOB sources *.cc)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUOgl++ TUV4L2++)
