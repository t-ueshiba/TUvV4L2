/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_LOOKUP_H)
#define __TU_SIMD_X86_LOOKUP_H

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

#  define SIMD_LOOKUP(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	typedef signed_type<upper_type<type> >	signed_upper_type;	\
	return cvt<type>(lookup(p, cvt<signed_upper_type, 0>(idx)),	\
			 lookup(p, cvt<signed_upper_type, 1>(idx)));	\
    }

  namespace detail
  {
    template <class S> static inline Is32vec
    lookup(const S* p, Is32vec idx, std::true_type)
    {
	constexpr size_t	n = sizeof(int32_t) - sizeof(S);
	const void*		q = (const int8_t*)p - n;
	return  _mm256_srai_epi32(_mm256_i32gather_epi32((const int32_t*)q,
							 idx, sizeof(S)), 8*n);
    }
    template <class S> static inline Is32vec
    lookup(const S* p, Is32vec idx, std::false_type)
    {
	constexpr size_t	n = sizeof(int32_t) - sizeof(S);
	const void*		q = (const int8_t*)p - n;
	return _mm256_srli_epi32(_mm256_i32gather_epi32((const int32_t*)q,
							idx, sizeof(S)), 8*n);
    }
  }
    
  template <class S> inline Is32vec
  lookup(const S* p, Is32vec idx)
  {
      return detail::lookup(p, idx, std::is_signed<S>());
  }


  SIMD_LOOKUP(int16_t)
  SIMD_LOOKUP(u_int16_t)
  SIMD_LOOKUP(int8_t)
  SIMD_LOOKUP(u_int8_t)

#  undef SIMD_LOOKUP

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
#endif	// !__TU_SIMD_X86_LOOKUP_H
