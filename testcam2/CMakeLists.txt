project(iidctestcam2)

include_directories(${PROJECT_SOURCE_DIR}/../testcam)
file(GLOB sources *.cc)
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} TUvIIDC++)
