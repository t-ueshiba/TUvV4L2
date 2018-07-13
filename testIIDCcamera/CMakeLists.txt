cmake_minimum_required(VERSION 2.4)

project(testIIDCcamera)

enable_language(CXX)
set(CMAKE_CXX_STANDARD		14)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)
add_definitions(-DNDEBUG)

find_package(GTK2 2.6 REQUIRED gtk)

file(GLOB sources *.cc)

include_directories(${GTK2_INCLUDE_DIRS})
add_executable(testIIDCcamera ${sources} ${headers})
target_link_libraries(testIIDCcamera TUIIDC++ TUTools++ ${GTK2_LIBRARIES})

install(TARGETS testIIDCcamera RUNTIME DESTINATION bin)

