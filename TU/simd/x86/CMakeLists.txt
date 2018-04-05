set(headers
  arch.h
  arithmetic.h
  bit_shift.h
  cast.h
  compare.h
  cvt.h
  dup.h
  insert_extract.h
  load_store.h
  logical.h
  logical_base.h
  lookup.h
  misc.h
  select.h
  shift.h
  shuffle.h
  svml.h
  type_traits.h
  unpack.h
  vec.h
  zero.h)

install(FILES ${headers} DESTINATION include/TU/simd/x86)

