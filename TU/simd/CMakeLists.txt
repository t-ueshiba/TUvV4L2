set(headers
  Array++.h
  allocator.h
  arithmetic.h
  assignment_iterator.h
  bit_shift.h
  cast.h
  compare.h
  config.h
  cvt.h
  cvtdown_iterator.h
  cvtup_iterator.h
  dup.h
  insert_extract.h
  iterator_wrapper.h
  load_store.h
  load_store_iterator.h
  logical.h
  lookup.h
  map_iterator.h
  misc.h
  select.h
  shift.h
  shift_iterator.h
  simd.h
  transform.h
  type_traits.h
  vec.h)

add_subdirectory(x86)
add_subdirectory(arm)

install(FILES ${headers} DESTINATION include/TU/simd)

