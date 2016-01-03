/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_LOOKUP_H)
#define __TU_SIMD_INTEL_LOOKUP_H

namespace TU
{
namespace simd
{
#if defined(AVX2)
#  define SIMD_LOOKUP32(from, to)					\
    SIMD_SPECIALIZED_FUNC(vec<to> lookup(const to* p, vec<from> idx),	\
			  i32gather,					\
			  ((const signed_type<to>*)p, idx, sizeof(to)), \
			  void, to, SIMD_SIGNED)
#  define SIMD_LOOKUP64(from, to)					\
    SIMD_SPECIALIZED_FUNC(vec<to> lookup(const to* p, vec<from> idx),	\
			  i64gather,					\
			  ((const signed_type<to>*)p, idx, sizeof(to)), \
			  void, to, SIMD_SIGNED)

  SIMD_LOOKUP32(int32_t,   int32_t)
  SIMD_LOOKUP32(u_int32_t, int32_t)
  SIMD_LOOKUP32(int32_t,   u_int32_t)
  SIMD_LOOKUP32(u_int32_t, u_int32_t)
  SIMD_LOOKUP32(int32_t,   float)
  SIMD_LOOKUP32(u_int32_t, float)

  SIMD_LOOKUP64(int64_t,   int64_t)
  SIMD_LOOKUP64(u_int64_t, int64_t)
  SIMD_LOOKUP64(int64_t,   u_int64_t)
  SIMD_LOOKUP64(u_int64_t, u_int64_t)
  SIMD_LOOKUP64(int64_t,   double)
  SIMD_LOOKUP64(u_int64_t, double)

#  undef SIMD_LOOKUP32
#  undef SIMD_LOOKUP64

template <class A, class S>
inline typename std::enable_if<vec<element_t<A> >::size == vec<S>::size,
			       vec<element_t<A> > >::type
lookup(const A& a, vec<S> row, vec<S> col)
{
    return lookup(a.data(), row * vec<S>(a.ncol()) + col);
}

#else	// !AVX2
namespace detail
{
  template <class T, class S, size_t... IDX> inline vec<T>
  lookup(const T* p, vec<S> idx, std::index_sequence<IDX...>)
  {
      return vec<T>(p[extract<IDX>(idx)]...);
  }

  template <class A, class S, size_t... IDX> inline vec<element_t<A> >
  lookup(const A& a, vec<S> row, vec<S> col, std::index_sequence<IDX...>)
  {
      return vec<element_t<A> >(a[extract<IDX>(row)][extract<IDX>(col)]...);
  }
}
    
template <class T, class S>
inline typename std::enable_if<vec<T>::size == vec<S>::size, vec<T> >::type
lookup(const T* p, vec<S> idx)
{
    return detail::lookup(p, idx, std::make_index_sequence<vec<S>::size>());
}

template <class A, class S>
inline typename std::enable_if<vec<element_t<A> >::size == vec<S>::size,
			       vec<element_t<A> > >::type
lookup(const A& a, vec<S> row, vec<S> col)
{
    return detail::lookup(a, row, col,
			  std::make_index_sequence<vec<S>::size>());
}

#endif
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_LOOKUP_H
