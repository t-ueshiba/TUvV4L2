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

  inline F32vec
  lookup(const float* p, Is32vec idx)
  {
      return _mm256_i32gather_ps(p, idx, sizeof(float));
  }

  inline F64vec
  lookup(const double* p, Is32vec idx)
  {
      return _mm256_i32gather_pd(p, _mm256_extractf128_si256(idx, 0x0),
				 sizeof(double));
  }
    
#else	// !AVX2
#  define SIMD_LOOKUP4(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract<0>(idx)], p[extract<1>(idx)],	\
			 p[extract<2>(idx)], p[extract<3>(idx)]);	\
    }
#  if defined(SSE2)
#    define SIMD_LOOKUP8(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract<0>(idx)], p[extract<1>(idx)],	\
			 p[extract<2>(idx)], p[extract<3>(idx)],	\
			 p[extract<4>(idx)], p[extract<5>(idx)],	\
			 p[extract<6>(idx)], p[extract<7>(idx)]);	\
    }
#  else		// !SSE2
#    define SIMD_LOOKUP8(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	const Is16vec	idx_lo = cvt<int16_t, 0>(idx),			\
			idx_hi = cvt<int16_t, 1>(idx);			\
	return vec<type>(p[extract<0>(idx_lo)], p[extract<1>(idx_lo)],	\
			 p[extract<2>(idx_lo)], p[extract<3>(idx_lo)],	\
			 p[extract<0>(idx_hi)], p[extract<1>(idx_hi)],	\
			 p[extract<2>(idx_hi)], p[extract<3>(idx_hi)]);	\
    }
#  endif
#  if defined(SSE4)
#    define SIMD_LOOKUP16(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	return vec<type>(p[extract< 0>(idx)], p[extract< 1>(idx)],	\
			 p[extract< 2>(idx)], p[extract< 3>(idx)],	\
			 p[extract< 4>(idx)], p[extract< 5>(idx)],	\
			 p[extract< 6>(idx)], p[extract< 7>(idx)],	\
			 p[extract< 8>(idx)], p[extract< 9>(idx)],	\
			 p[extract<10>(idx)], p[extract<11>(idx)],	\
			 p[extract<12>(idx)], p[extract<13>(idx)],	\
			 p[extract<14>(idx)], p[extract<15>(idx)]);	\
    }
#  else		// !SSE4
#    define SIMD_LOOKUP16(type)						\
    template <class S> inline vec<type>					\
    lookup(const S* p, vec<type> idx)					\
    {									\
	const Is16vec	idx_lo = cvt<int16_t, 0>(idx),			\
			idx_hi = cvt<int16_t, 1>(idx);			\
	return vec<type>(p[extract<0>(idx_lo)], p[extract<1>(idx_lo)],	\
			 p[extract<2>(idx_lo)], p[extract<3>(idx_lo)],	\
			 p[extract<4>(idx_lo)], p[extract<5>(idx_lo)],	\
			 p[extract<6>(idx_lo)], p[extract<7>(idx_lo)],	\
			 p[extract<0>(idx_hi)], p[extract<1>(idx_hi)],	\
			 p[extract<2>(idx_hi)], p[extract<3>(idx_hi)],	\
			 p[extract<4>(idx_hi)], p[extract<5>(idx_hi)],	\
			 p[extract<6>(idx_hi)], p[extract<7>(idx_hi)]);	\
    }
#  endif

#  if defined(SSE2)
  SIMD_LOOKUP16(int8_t)
  SIMD_LOOKUP16(u_int8_t)
  SIMD_LOOKUP8(int16_t)
  SIMD_LOOKUP8(u_int16_t)
#    if defined(SSE4)
    SIMD_LOOKUP4(u_int32_t)
    SIMD_LOOKUP4(int32_t)
#    endif
#  else		// !SSE2
  SIMD_LOOKUP8(int8_t)
  SIMD_LOOKUP8(u_int8_t)
  SIMD_LOOKUP4(int16_t)
  SIMD_LOOKUP4(u_int16_t)
#  endif
#  undef SIMD_LOOKUP4
#  undef SIMD_LOOKUP8
#  undef SIMD_LOOKUP16
#endif
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_LOOKUP_H
