project(raw1394)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

include_directories(${PROJECT_SOURCE_DIR})

file(GLOB sources *.cc)

add_library(${PROJECT_NAME} SHARED ${sources})
target_link_libraries(${PROJECT_NAME} "-framework IOKit -framework CoreFoundation -framework CoreServices")

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

add_subdirectory(libraw1394)
