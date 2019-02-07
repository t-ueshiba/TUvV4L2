cmake_minimum_required(VERSION 2.4)
project(testIIDCcamera2)

enable_language(CXX)
set(CMAKE_CXX_STANDARD		14)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

find_package(Qt5Widgets REQUIRED)
include_directories(${Qt5Widgets_INCLUDE_DIRS})

file(GLOB sources *.cc)

set(moc_headers CmdPane.h SliderCmd.h)
set(moc_sources "")
QT5_WRAP_CPP(moc_sources ${moc_headers})

add_executable(${PROJECT_NAME} ${sources} ${moc_sources})
target_link_libraries(${PROJECT_NAME} TUIIDC++ ${Qt5Widgets_LIBRARIES})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

