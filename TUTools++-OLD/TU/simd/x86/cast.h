/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_CAST_H)
#define __TU_SIMD_X86_CAST_H

namespace TU
{
namespace simd
{
namespace detail
{
  template <class T, class S> T	cast_base(S x)				;

  /*
   *  cast_base() の実装
   */
  // 整数 <-> 整数
  template <> inline ivec_t
  cast_base(ivec_t x)
  {
      return x;
  }

// 整数 <-> float, double
#if !defined(AVX2) && defined(AVX)
  template <> fvec_t
  cast_base<fvec_t>(ivec_t x)
  {
      return _mm256_castsi256_ps(_mm256_castsi128_si256(x));
  }

  template <> ivec_t
  cast_base<ivec_t>(fvec_t x)
  {
      return _mm256_castsi256_si128(_mm256_castps_si256(x));
  }

  template <> dvec_t
  cast_base<dvec_t>(ivec_t x)
  {
      return _mm256_castsi256_pd(_mm256_castsi128_si256(x));
  }

  template <> ivec_t
  cast_base<ivec_t>(dvec_t x)
  {
      return _mm256_castsi256_si128(_mm256_castpd_si256(x));
  }
#elif defined(SSE2)
  SIMD_SPECIALIZED_FUNC(fvec_t cast_base<fvec_t>(ivec_t x),
			cast, (x), ivec_t, fvec_t, SIMD_BASE)
  SIMD_SPECIALIZED_FUNC(ivec_t cast_base<ivec_t>(fvec_t x),
			cast, (x), fvec_t, ivec_t, SIMD_BASE)
  SIMD_SPECIALIZED_FUNC(dvec_t cast_base<dvec_t>(ivec_t x),
			cast, (x), ivec_t, dvec_t, SIMD_BASE)
  SIMD_SPECIALIZED_FUNC(ivec_t cast_base<ivec_t>(dvec_t x),
			cast, (x), dvec_t, ivec_t, SIMD_BASE)
#endif

// float <-> double
#if defined(SSE2)
  SIMD_SPECIALIZED_FUNC(dvec_t cast_base<dvec_t>(fvec_t x),
			cast, (x), fvec_t, dvec_t, SIMD_BASE)
  SIMD_SPECIALIZED_FUNC(fvec_t cast_base<fvec_t>(dvec_t x),
			cast, (x), dvec_t, fvec_t, SIMD_BASE)
#endif
}	// namespace detail
      
template <class T, class S> inline vec<T>
cast(vec<S> x)
{
    return detail::cast_base<base_type<T> >(base_type<S>(x));
}
    
}	// namespace simd
}	// namespace TU

#endif	// !__TU_SIMD_X86_CAST_H
