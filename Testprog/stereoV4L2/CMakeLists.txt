project(stereoV4L2)

set(EXTERNAL_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../stereoGUI)

include_directories(${EXTERNAL_SOURCE_DIR})
add_executable(
    ${PROJECT_NAME}
    main.cc
    ${EXTERNAL_SOURCE_DIR}/ComputeThreeD.cc
    ${EXTERNAL_SOURCE_DIR}/DrawThreeD.cc
    ${EXTERNAL_SOURCE_DIR}/MyCanvasPane.cc
)
target_link_libraries(${PROJECT_NAME} TUOgl++ TUvV4L2++)
