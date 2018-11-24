cmake_minimum_required(VERSION 2.4)

project(TUIIDC++)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

enable_language(CXX)
set(CMAKE_CXX_STANDARD		14)
set(CMAKE_CXX_STANDARD_REQUIRED	ON)
set(CMAKE_CXX_EXTENSIONS	OFF)
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)
add_definitions(-DNDEBUG)

find_package(PkgConfig)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)
find_package(Boost REQUIRED)

add_definitions(-DNDEBUG)

include_directories(
    .
    ${CMAKE_INSTALL_PREFIX}/include
    ${Boost_INCLUDE_DIRS}
    ${LIBUSB_INCLUDE_DIRS})
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB sources *.cc)

add_library(${PROJECT_NAME} SHARED ${sources})

if(APPLE)
    add_subdirectory(MacOSX)
    include_directories(MacOSX)
    target_link_libraries(${PROJECT_NAME} TUTools++ raw1394 ${LIBUSB_LIBRARIES})
else()
    pkg_check_modules(LIBRAW1394 REQUIRED libraw1394)
    include_directories(${LIBRAW1394_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} TUTools++
			  ${LIBRAW1394_LIBRARIES} ${LIBUSB_LIBRARIES})
endif()

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(TU)
