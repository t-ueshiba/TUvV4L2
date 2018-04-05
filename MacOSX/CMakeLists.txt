#
include_directories(. libraw1394)

set(sources
    raw1394.cc)

set(headers
    raw1394_.h)
    
if (COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif ()

add_library(raw1394 SHARED ${sources})
target_link_libraries(raw1394 "-framework IOKit -framework CoreFoundation -framework CoreServices")

set(serial "1.0.0")
set(soserial "1")
set_target_properties(raw1394	PROPERTIES
		      VERSION	${serial}
		      SOVERSION ${soserial})

install(TARGETS raw1394 LIBRARY DESTINATION lib)

add_subdirectory(libraw1394)
