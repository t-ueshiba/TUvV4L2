/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_LOAD_STORE_H)
#define __TU_SIMD_X86_LOAD_STORE_H

namespace TU
{
namespace simd
{
#if defined(SSE2)
#  define SIMD_LOAD_STORE(type)						\
    SIMD_SPECIALIZED_FUNC(vec<type> load<true>(const type* p),		\
			  load, ((const vec<type>::base_type*)p),	\
			  void, type, SIMD_BASE)			\
    SIMD_SPECIALIZED_FUNC(vec<type> load<false>(const type* p),		\
			  loadu, ((const vec<type>::base_type*)p),	\
			  void, type, SIMD_BASE)			\
    SIMD_SPECIALIZED_FUNC(void store<true>(type* p, vec<type> x),	\
			  store, ((vec<type>::base_type*)p, x),		\
			  void, type, SIMD_BASE)			\
    SIMD_SPECIALIZED_FUNC(void store<false>(type* p, vec<type> x),	\
			  storeu, ((vec<type>::base_type*)p, x),	\
			  void, type, SIMD_BASE)

  SIMD_LOAD_STORE(int8_t)
  SIMD_LOAD_STORE(int16_t)
  SIMD_LOAD_STORE(int32_t)
  SIMD_LOAD_STORE(int64_t)
  SIMD_LOAD_STORE(uint8_t)
  SIMD_LOAD_STORE(uint16_t)
  SIMD_LOAD_STORE(uint32_t)
  SIMD_LOAD_STORE(uint64_t)

#  undef SIMD_LOAD_STORE  
#else
  template <bool ALIGNED, class T> inline vec<T>
  load(const T* p)
  {
      return *((const typename vec<T>::base_type*)p);
  }
  template <bool ALIGNED, class T> inline void
  store(T* p, vec<T> x)
  {
      *((typename vec<T>::base_type*)p) = x;
  }
#endif

#if defined(SSE)
#  define SIMD_LOAD_STORE(type)						\
    SIMD_SPECIALIZED_FUNC(vec<type> load<true>(const type* p),		\
			  load, (p), void, type, SIMD_BASE)		\
    SIMD_SPECIALIZED_FUNC(vec<type> load<false>(const type* p),		\
			  loadu, (p), void, type, SIMD_BASE)		\
    SIMD_SPECIALIZED_FUNC(void store<true>(type* p, vec<type> x),	\
			  store, (p, x), void, type, SIMD_BASE)		\
    SIMD_SPECIALIZED_FUNC(void store<false>(type* p, vec<type> x),	\
			  storeu, (p, x), void, type, SIMD_BASE)

  SIMD_LOAD_STORE(float)
#  if defined(SSE2)
  SIMD_LOAD_STORE(double)
#  endif
#  undef SIMD_LOAD_STORE
#endif

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_LOAD_STORE_H
