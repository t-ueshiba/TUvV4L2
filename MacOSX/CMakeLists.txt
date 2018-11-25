<<<<<<< HEAD
include_directories(. libraw1394)

set(sources
    raw1394.cc)
=======
project(raw1394)
>>>>>>> 3b2e6079461072eee66a839f4e3b282b68b4277b

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

<<<<<<< HEAD
add_library(raw1394 SHARED ${sources})
target_link_libraries(raw1394 "-framework IOKit -framework CoreFoundation -framework CoreServices")

install(TARGETS raw1394 LIBRARY DESTINATION lib)
=======
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB sources *.cc)

add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME} "-framework IOKit -framework CoreFoundation -framework CoreServices")

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
>>>>>>> 3b2e6079461072eee66a839f4e3b282b68b4277b

add_subdirectory(libraw1394)
