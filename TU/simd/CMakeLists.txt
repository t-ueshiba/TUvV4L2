file(GLOB headers *.h)

add_subdirectory(x86)
add_subdirectory(arm)

install(FILES ${headers} DESTINATION include/TU/simd)

