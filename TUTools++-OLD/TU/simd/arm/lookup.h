/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_LOOKUP_H)
#define __TU_SIMD_ARM_LOOKUP_H

namespace TU
{
namespace simd
{
namespace detail
{
  template <class P, class T, size_t... IDX>
  inline typename std::enable_if<(vec<P>::size == vec<T>::size), vec<P> >::type
  lookup(const P* p, vec<T> idx, std::index_sequence<IDX...>)
  {
      return vec<P>(p[extract<IDX>(idx)]...);
  }

  template <class P, class T, size_t... IDX>
  inline typename std::enable_if<(vec<P>::size != vec<T>::size), vec<T> >::type
  lookup(const P* p, vec<T> idx, std::index_sequence<IDX...>)
  {
      return vec<T>(p[extract<IDX>(idx)]...);
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

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_LOOKUP_H
