file(GLOB headers *.h)

add_subdirectory(simd)

install(FILES ${headers} DESTINATION include/TU)
