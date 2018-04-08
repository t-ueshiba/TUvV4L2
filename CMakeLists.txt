#
#
#
cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

find_package(PkgConfig)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

include_directories(
    .
    ../TUTools++
    ${LIBUSB_INCLUDE_DIRS})

set(sources
    FireWireNode.cc
    IIDCCamera.cc
    IIDCCameraArray.cc
    IIDCNode.cc
    USBNode.cc)

if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif ()

add_library(TUIIDC++ SHARED ${sources})

if (APPLE)
  add_subdirectory(MacOSX)
  include_directories(MacOSX)
  target_link_libraries(
      TUIIDC++
      TUTools++
      raw1394
      ${LIBUSB_LIBRARIES})
else ()
  pkg_check_modules(LIBRAW1394 REQUIRED libraw1394)
  include_directories(${LIBRAW1394_INCLUDE_DIRS})
  target_link_libraries(
      TUIIDC++
      TUTools++
      ${LIBRAW1394_LIBRARIES}
      ${LIBUSB_LIBRARIES})
endif ()

set(serial "1.0.0")
set(soserial "1")
set_target_properties(TUIIDC++	PROPERTIES
		      VERSION	${serial}
		      SOVERSION ${soserial})

install(TARGETS TUIIDC++ LIBRARY DESTINATION lib)

add_subdirectory(TU)
