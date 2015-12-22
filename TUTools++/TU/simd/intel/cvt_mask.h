/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_CVT_MASK_H)
#define __TU_SIMD_INTEL_CVT_MASK_H

#include "TU/simd/intel/cast.h"

namespace TU
{
namespace simd
{
// [1] 整数ベクトル間のマスク変換
#if defined(AVX2)
#  define SIMD_CVTUP_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask<to, 0>(vec<from> x)					\
    {									\
	return SIMD_MNEMONIC(cvt, _mm256_,				\
			     SIMD_SIGNED(from), SIMD_SIGNED(to))(	\
				 _mm256_castsi256_si128(x));		\
    }									\
    template <> inline vec<to>						\
    cvt_mask<to, 1>(vec<from> x)					\
    {									\
	return SIMD_MNEMONIC(cvt, _mm256_,				\
			     SIMD_SIGNED(from), SIMD_SIGNED(to))(	\
				 _mm256_extractf128_si256(x, 0x1));	\
    }
#  define SIMD_CVTDOWN_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask<to>(vec<from> x, vec<from> y)				\
    {									\
	return SIMD_MNEMONIC(packs, _mm256_, , SIMD_SIGNED(from))(	\
	    _mm256_permute2f128_si256(x, y, 0x20),			\
	    _mm256_permute2f128_si256(x, y, 0x31));			\
    }
#else
#  define SIMD_CVTUP_MASK(from, to)					\
    template <> inline vec<to>						\
    cvt_mask<to, 0>(vec<from> x)					\
    {									\
	return SIMD_MNEMONIC(unpacklo,					\
			     _mm_, , SIMD_SIGNED(from))(x, x);		\
    }									\
    template <> inline vec<to>						\
    cvt_mask<to, 1>(vec<from> x)					\
    {									\
	return SIMD_MNEMONIC(unpackhi,					\
			     _mm_, , SIMD_SIGNED(from))(x, x);		\
    }
#  define SIMD_CVTDOWN_MASK(from, to)					\
    SIMD_SPECIALIZED_FUNC(vec<to> cvt_mask<to>(vec<from> x, vec<from> y), \
			  packs, (x, y), void, from, SIMD_SIGNED)
#endif
#define SIMD_CVT_MASK(type0, type1)					\
    SIMD_CVTUP_MASK(type0, type1)					\
    SIMD_CVTDOWN_MASK(type1, type0)

SIMD_CVT_MASK(u_int8_t,	   u_int16_t)	// u_char  <-> u_short
SIMD_CVT_MASK(u_int16_t,   u_int32_t)	// u_short <-> u_int
SIMD_CVTUP_MASK(u_int32_t, u_int64_t)	// u_int    -> u_long

#undef SIMD_CVTUP_MASK
#undef SIMD_CVTDOWN_MASK
#undef SIMD_CVT_MASK

// [2] 整数ベクトルと浮動小数点数ベクトル間のマスク変換
#if defined(SSE2)
#  if !defined(AVX) || defined(AVX2)	// Is32vec::size == F32vec::size
#    define SIMD_CVT_MASK_2FI(itype, ftype)				\
      template <> inline vec<ftype>					\
      cvt_mask<ftype>(vec<itype> x)	{return cast<ftype>(x);}	\
      template <> inline vec<itype>					\
      cvt_mask<itype>(vec<ftype> x)	{return cast<itype>(x);}

    SIMD_CVT_MASK_2FI(u_int32_t, float)		// u_int  <-> float
    SIMD_CVT_MASK_2FI(u_int64_t, double)	// u_long <-> double

#    undef SIMD_CVT_MASK_2FI
#  else	// AVX && !AVX2
#    define SIMD_CVT_MASK_IF(itype, ftype)				\
      template <> inline vec<ftype>					\
      cvt_mask<ftype>(vec<itype> x)					\
      {									\
	  typedef upper_type<itype>	upper_type;			\
      									\
	  return SIMD_MNEMONIC(cast,					\
			       _mm256_, si256, SIMD_SUFFIX(ftype))(	\
				   _mm256_insertf128_si256(		\
				       _mm256_castsi128_si256(		\
					   cvt_mask<upper_type, 0>(x)),	\
				       cvt_mask<upper_type, 1>(x),	\
					   0x1));			\
      }
#    define SIMD_CVT_MASK_FI(itype)					\
      template <> inline vec<itype>					\
      cvt_mask<itype>(F32vec x)						\
      {									\
	  typedef upper_type<itype>	upper_type;			\
	  								\
	  return cvt_mask<itype>(					\
		     vec<upper_type>(					\
			 _mm256_castsi256_si128(			\
			     _mm256_castps_si256(x))),			\
		     vec<upper_type>(					\
		         _mm256_extractf128_si256(			\
			     _mm256_castps_si256(x), 0x1)));		\
      }

    SIMD_CVT_MASK_IF(u_int16_t, float)		// u_short -> float
    SIMD_CVT_MASK_FI(u_int16_t)			// float   -> u_short
    SIMD_CVT_MASK_IF(u_int32_t, double)		// u_int   -> double

#    undef SIMD_CVT_MASK_IF
#    undef SIMD_CVT_MASK_FI
#  endif
#endif
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_CVT_MASK_H
