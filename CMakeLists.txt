cmake_minimum_required(VERSION 2.4)
project(TUIIDC++)

set(CMAKE_CXX_FLAGS		"${CMAKE_CXX_FLAGS} -msse4 -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE	"-Wall -O3")
set(CMAKE_CXX_FLAGS_DEBUG	"-g")
set(CMAKE_BUILD_TYPE		Release)

add_definitions(-DNDEBUG -DSSE4)

find_package(PkgConfig)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

include_directories(${LIBUSB_INCLUDE_DIRS})

set(sources
    FireWireNode.cc
    IIDCCamera.cc
    IIDCCameraArray.cc
    IIDCNode.cc
    USBNode.cc)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif()

add_library(TUIIDC++ SHARED ${sources})

if(APPLE)
    add_subdirectory(MacOSX)
    include_directories(MacOSX)
    target_link_libraries(TUIIDC++ TUTools++ raw1394 ${LIBUSB_LIBRARIES})
else()
    pkg_check_modules(LIBRAW1394 REQUIRED libraw1394)
    include_directories(${LIBRAW1394_INCLUDE_DIRS})
    target_link_libraries(TUIIDC++ TUTools++
			  ${LIBRAW1394_LIBRARIES} ${LIBUSB_LIBRARIES})
endif()

install(TARGETS TUIIDC++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
