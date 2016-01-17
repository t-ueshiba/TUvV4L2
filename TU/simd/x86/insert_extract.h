/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_INSERT_EXTRACT_H)
#define __TU_SIMD_X86_INSERT_EXTRACT_H

namespace TU
{
namespace simd
{
#if defined(SSE)
#  if defined(AVX2)
#    define SIMD_INSERT_EXTRACT(type)					\
      template <size_t I> inline vec<type>				\
      insert(vec<type> x, int val)					\
      {									\
	  return _mm256_insertf128_si256(				\
		     x,							\
		     SIMD_MNEMONIC(insert, _mm_, , SIMD_SIGNED(type))(	\
			 _mm256_extractf128_si256(			\
			     x, I / vec<type>::lane_size),		\
			 val, I % vec<type>::lane_size),		\
		     I / vec<type>::lane_size);				\
      }									\
      template <size_t I> inline int					\
      extract(vec<type> x)						\
      {									\
	  return SIMD_MNEMONIC(extract, _mm_, , SIMD_SIGNED(type))(	\
		     _mm256_extractf128_si256(				\
			 x, I / vec<type>::lane_size),			\
		     I % vec<type>::lane_size);				\
      }

#  else
#    define SIMD_INSERT_EXTRACT(type)					\
      template <size_t I>		 				\
      SIMD_FUNC(vec<type> insert(vec<type> x, int val), insert,		\
		(x, val, I), void, type, SIMD_SIGNED)			\
      template <size_t I>						\
      SIMD_FUNC(int extract(vec<type> x), extract,			\
		(x, I), void, type, SIMD_SIGNED)
#  endif

  SIMD_INSERT_EXTRACT(int16_t)
  SIMD_INSERT_EXTRACT(u_int16_t)
#  if defined(SSE4)
    SIMD_INSERT_EXTRACT(int8_t)
    SIMD_INSERT_EXTRACT(u_int8_t)
    SIMD_INSERT_EXTRACT(int32_t)
    SIMD_INSERT_EXTRACT(u_int32_t)
#  endif

#  undef SIMD_INSERT_EXTRACT

#  if defined(AVX)
    template <size_t I> inline F32vec
    insert(F32vec x, float val)
    {
	return _mm256_insertf128_ps(x,
				    _mm_insert_ps(
					_mm256_extractf128_ps(
					    x, I / F32vec::lane_size),
					_mm_set_ss(val),
					(I % F32vec::lane_size) << 4),
				    I / F32vec::lane_size);
    }
#  elif defined(SSE4)
    template <size_t I> inline F32vec
    insert(F32vec x, float val)
    {
	return _mm_insert_ps(x, _mm_set_ss(val), I << 4);
    }
#  endif

template <size_t I> float	extract(F32vec x)			;
#  if defined(AVX)
    template <> inline float
    extract<0>(F32vec x)
    {
	return _mm_cvtss_f32(_mm256_extractf128_ps(x, 0x0));
    }

    template <> inline float
    extract<4>(F32vec x)
    {
	return _mm_cvtss_f32(_mm256_extractf128_ps(x, 0x1));
    }
#  else
    template <> inline float
    extract<0>(F32vec x)
    {
	return _mm_cvtss_f32(x);
    }
#  endif

#  if defined(SSE2)
    template <size_t I> double	extract(F64vec x)			;
#    if defined(AVX)
      template <> inline double
      extract<0>(F64vec x)
      {
	  return _mm_cvtsd_f64(_mm256_extractf128_pd(x, 0x0));
      }

      template <> inline double
      extract<2>(F64vec x)
      {
	  return _mm_cvtsd_f64(_mm256_extractf128_pd(x, 0x1));
      }
#    else
      template <> inline double
      extract<0>(F64vec x)
      {
	  return _mm_cvtsd_f64(x);
      }
#    endif
#  endif
#endif  
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_INSERT_EXTRACT_H
