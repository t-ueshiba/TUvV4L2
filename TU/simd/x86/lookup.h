/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_LOOKUP_H)
#define TU_SIMD_X86_LOOKUP_H

namespace TU
{
namespace simd
{
#if defined(AVX2)
#  define SIMD_LOOKUP32(to)						\
    SIMD_SPECIALIZED_FUNC(vec<to> lookup(const to* p, Is32vec idx),	\
			  i32gather,					\
			  (reinterpret_cast<const signed_type<to>*>(p),	\
			   idx, sizeof(to)), void, to, SIMD_SIGNED)
#  define SIMD_LOOKUP64(to)						\
    SIMD_SPECIALIZED_FUNC(vec<to> lookup(const to* p, Is64vec idx),	\
			  i64gather,					\
			  (reinterpret_cast<const signed_type<to>*>(p), \
			   idx, sizeof(to)), void, to, SIMD_SIGNED)

  SIMD_LOOKUP32(int32_t)
  SIMD_LOOKUP32(uint32_t)
  SIMD_LOOKUP32(float)

//SIMD_LOOKUP64(int64_t)
//SIMD_LOOKUP64(uint64_t)
  SIMD_LOOKUP64(double)

#  undef SIMD_LOOKUP32
#  undef SIMD_LOOKUP64

  namespace detail
  {
    template <class P> inline Is32vec
    lookup(const P* p, Is32vec idx, std::true_type)
    {
        constexpr int	N = sizeof(int32_t)/sizeof(P) - 1;

        return  _mm256_srai_epi32(
		    _mm256_i32gather_epi32(
			reinterpret_cast<const int32_t*>(p - N),
			idx, sizeof(P)),
		    8*sizeof(P)*N);
    }
    template <class P> inline Is32vec
    lookup(const P* p, Is32vec idx, std::false_type)
    {
	constexpr int	N = sizeof(int32_t)/sizeof(P) - 1;

	return  _mm256_srli_epi32(
		    _mm256_i32gather_epi32(
			reinterpret_cast<const int32_t*>(p - N),
			idx, sizeof(P)),
		    8*sizeof(P)*N);
    }
  }	// namespace detail

  template <class P>
  inline typename std::enable_if<(vec<P>::size > Is32vec::size), Is32vec>::type
  lookup(const P* p, Is32vec idx)
  {
      return detail::lookup(p, idx, std::is_signed<P>());
  }

  template <class P, class T,
	    typename std::enable_if<(vec<P>::size == vec<T>::size)>::type*>
  inline vec<P>
  lookup(const P* p, vec<T> idx)
  {
      using U = signed_type<upper_type<T> >;
      
      return cvt<P>(lookup(p, cvt<U, false>(idx)),
		    lookup(p, cvt<U, true >(idx)));
  }

  template <class P, class T,
	    typename std::enable_if<(vec<P>::size > vec<T>::size)>::type*>
  inline vec<T>
  lookup(const P* p, vec<T> idx)
  {
      using U = signed_type<upper_type<T> >;
      
      return cvt<T>(lookup(p, cvt<U, false>(idx)),
		    lookup(p, cvt<U, true >(idx)));
  }

#else	// !AVX2
  namespace detail
  {
    template <class P, class T, size_t... IDX>
    inline typename std::enable_if<(vec<P>::size == vec<T>::size),
				   vec<P> >::type
    lookup(const P* p, vec<T> idx, std::index_sequence<IDX...>)
    {
	return vec<P>(p[extract<IDX>(idx)]...);
    }

    template <class P, class T, size_t... IDX>
    inline typename std::enable_if<(vec<P>::size != vec<T>::size),
				   vec<T> >::type
    lookup(const P* p, vec<T> idx, std::index_sequence<IDX...>)
    {
	return vec<T>(p[extract<IDX>(idx)]...);
    }

    template <class P, size_t... IDX> inline auto
    lookup(const P* p, Is16vec row, Is16vec col, int32_t stride,
	   std::index_sequence<IDX...>) -> decltype(lookup(p, col))
    {
	return {p[extract<IDX>(row)*stride + extract<IDX>(col)]...};
    }
  }	// namespace detail
    
  template <class P, class T,
	    typename std::enable_if<(vec<P>::size == vec<T>::size)>::type*>
  inline vec<P>
  lookup(const P* p, vec<T> idx)
  {
      return detail::lookup(p, idx, std::make_index_sequence<vec<T>::size>());
  }

  template <class P, class T,
	    typename std::enable_if<(vec<P>::size > vec<T>::size)>::type*>
  inline vec<T>
  lookup(const P* p, vec<T> idx)
  {
      return detail::lookup(p, idx, std::make_index_sequence<vec<T>::size>());
  }

#endif

#if defined(SSE4)
  template <class P> inline auto
  lookup(const P* p, Is16vec row, Is16vec col, int32_t stride)
      -> decltype(lookup(p, col))
  {
      using T = typename decltype(lookup(p, col))::element_type;

      return cvt<T>(lookup(p,
			   cvt<int32_t, false>(row)*stride +
			   cvt<int32_t, false>(col)),
		    lookup(p,
			   cvt<int32_t, true >(row)*stride +
			   cvt<int32_t, true >(col)));
  }
#else
  template <class P> inline auto
  lookup(const P* p, Is16vec row, Is16vec col, int32_t stride)
      -> decltype(lookup(p, col))
  {
      return detail::lookup(p, row, col, stride,
			    std::make_index_sequence<Is16vec::size>());
  }
#endif
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_LOOKUP_H
