/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_CVT_H)
#define __TU_SIMD_INTEL_CVT_H

#include "TU/simd/intel/dup.h"

namespace TU
{
namespace simd
{
// [1] 整数ベクトル間の変換
#if defined(SSE4)
#  if defined(AVX2)
#    define SIMD_CVTUP0(from, to)					\
      template <> inline vec<to>					\
      cvt<to, 0>(vec<from> x)						\
      {									\
	  return SIMD_MNEMONIC(cvt, _mm256_, SIMD_SUFFIX(from),		\
			       SIMD_SIGNED(to))				\
	      (_mm256_castsi256_si128(x));				\
      }
#    define SIMD_CVTUP1(from, to)					\
      template <> inline vec<to>					\
      cvt<to, 1>(vec<from> x)						\
      {									\
	  return SIMD_MNEMONIC(cvt, _mm256_, SIMD_SUFFIX(from),		\
			       SIMD_SIGNED(to))				\
	      (_mm256_extractf128_si256(x, 0x1));			\
      }
#  else	// SSE4 && !AVX2
#    define SIMD_CVTUP0(from, to)					\
      template <> inline vec<to>					\
      cvt<to, 0>(vec<from> x)						\
      {									\
	  return SIMD_MNEMONIC(cvt, _mm_,				\
			       SIMD_SUFFIX(from), SIMD_SIGNED(to))(x);	\
      }
#    define SIMD_CVTUP1(from, to)					\
      template <> inline vec<to>					\
      cvt<to, 1>(vec<from> x)						\
      {									\
	  return cvt<to>(shift_r<vec<from>::size/2>(x));		\
      }
#  endif
  SIMD_CVTUP0(int8_t,    int16_t)	// s_char -> short
  SIMD_CVTUP1(int8_t,    int16_t)	// s_char -> short
  SIMD_CVTUP0(int8_t,    int32_t)	// s_char -> int
  SIMD_CVTUP0(int8_t,    int64_t)	// s_char -> long
  
  SIMD_CVTUP0(int16_t,   int32_t)	// short  -> int
  SIMD_CVTUP1(int16_t,   int32_t)	// short  -> int
  SIMD_CVTUP0(int16_t,   int64_t)	// short  -> long
  
  SIMD_CVTUP0(int32_t,   int64_t)	// int    -> long
  SIMD_CVTUP1(int32_t,   int64_t)	// int    -> long

  SIMD_CVTUP0(u_int8_t,  int16_t)	// u_char -> short
  SIMD_CVTUP1(u_int8_t,  int16_t)	// u_char -> short
  SIMD_CVTUP0(u_int8_t,  u_int16_t)	// u_char -> u_short
  SIMD_CVTUP1(u_int8_t,  u_int16_t)	// u_char -> u_short
  SIMD_CVTUP0(u_int8_t,  int32_t)	// u_char -> int
  SIMD_CVTUP0(u_int8_t,  u_int32_t)	// u_char -> u_int
  SIMD_CVTUP0(u_int8_t,  int64_t)	// u_char -> long
  SIMD_CVTUP0(u_int8_t,  u_int64_t)	// u_char -> u_long
  
  SIMD_CVTUP0(u_int16_t, int32_t)	// u_short -> int
  SIMD_CVTUP1(u_int16_t, int32_t)	// u_short -> int
  SIMD_CVTUP0(u_int16_t, u_int32_t)	// u_short -> u_int
  SIMD_CVTUP1(u_int16_t, u_int32_t)	// u_short -> u_int
  SIMD_CVTUP0(u_int16_t, int64_t)	// u_short -> long
  SIMD_CVTUP0(u_int16_t, u_int64_t)	// u_short -> u_long
  
  SIMD_CVTUP0(u_int32_t, int64_t)	// u_int -> long
  SIMD_CVTUP1(u_int32_t, int64_t)	// u_int -> long
  SIMD_CVTUP0(u_int32_t, u_int64_t)	// u_int -> u_long
  SIMD_CVTUP1(u_int32_t, u_int64_t)	// u_int -> u_long

#  undef SIMD_CVTUP0
#  undef SIMF_CVTUP1
#else	// !SSE4
#  define SIMD_CVTUP_I(from, to)					\
    template <> inline vec<to>						\
    cvt<to, 0>(vec<from> x)						\
    {									\
	return cast<to>(dup<0>(x)) >> 8*vec<from>::element_size;	\
    }									\
    template <> inline vec<to>						\
    cvt<to, 1>(vec<from> x)						\
    {									\
	return cast<to>(dup<1>(x)) >> 8*vec<from>::element_size;	\
    }
#  define SIMD_CVTUP_UI(from, to)					\
    template <> inline vec<to>						\
    cvt<to, 0>(vec<from> x)						\
    {									\
	return cast<to>(unpack_low(x, zero<from>()));			\
    }									\
    template <> inline vec<to>						\
    cvt<to, 1>(vec<from> x)						\
    {									\
	return cast<to>(unpack_high(x, zero<from>()));			\
    }

  SIMD_CVTUP_I(int8_t,     int16_t)	// s_char  -> short
  SIMD_CVTUP_I(int16_t,    int32_t)	// short   -> int
  // epi64の算術右シフトが未サポートなので int -> long は実装できない

  SIMD_CVTUP_UI(u_int8_t,  int16_t)	// u_char  -> short
  SIMD_CVTUP_UI(u_int8_t,  u_int16_t)	// u_char  -> u_short
  SIMD_CVTUP_UI(u_int16_t, int32_t)	// u_short -> int
  SIMD_CVTUP_UI(u_int16_t, u_int32_t)	// u_short -> u_int
  SIMD_CVTUP_UI(u_int32_t, int64_t)	// u_int   -> long
  SIMD_CVTUP_UI(u_int32_t, u_int64_t)	// u_int   -> u_long

#  undef SIMD_CVTUP_I
#  undef SIMD_CVTUP_UI
#endif

#if defined(AVX2)
#  define SIMD_CVTDOWN_I(from, to)					\
    template <> inline vec<to>						\
    cvt(vec<from> x, vec<from> y)					\
    {									\
	return SIMD_MNEMONIC(packs, _mm256_, , SIMD_SUFFIX(from))	\
	    (_mm256_permute2f128_si256(x, y, 0x20),			\
	     _mm256_permute2f128_si256(x, y, 0x31));			\
    }
#  define SIMD_CVTDOWN_UI(from, to)					\
    template <> inline vec<to>						\
    cvt(vec<from> x, vec<from> y)					\
    {									\
	return SIMD_MNEMONIC(packus, _mm256_, , SIMD_SUFFIX(from))	\
	    (_mm256_permute2f128_si256(x, y, 0x20),			\
	     _mm256_permute2f128_si256(x, y, 0x31));			\
    }
#else
#  define SIMD_CVTDOWN_I(from, to)					\
    SIMD_SPECIALIZED_FUNC(vec<to> cvt<to>(vec<from> x, vec<from> y),	\
			  packs, (x, y), void, from, SIMD_SIGNED)
#  define SIMD_CVTDOWN_UI(from, to)					\
    SIMD_SPECIALIZED_FUNC(vec<to> cvt<to>(vec<from> x, vec<from> y),	\
			  packus, (x, y), void, from, SIMD_SIGNED)
#endif

#define _mm_packus_pi16	_mm_packs_pu16	// 不適切な命名をSSE2に合わせて修正

SIMD_CVTDOWN_I(int16_t,  int8_t)	// short -> s_char
SIMD_CVTDOWN_I(int32_t,  int16_t)	// int   -> short
SIMD_CVTDOWN_UI(int16_t, u_int8_t)	// short -> u_char
#if defined(SSE4)
  SIMD_CVTDOWN_UI(int32_t, u_int16_t)	// int -> u_short
#endif

#undef SIMD_CVTDOWN_I
#undef SIMD_CVTDOWN_UI

// [2] 整数ベクトルと浮動小数点数ベクトル間の変換
#define SIMD_CVT(from, to)						\
  SIMD_SPECIALIZED_FUNC(vec<to> cvt<to>(vec<from> x),			\
			cvt, (x), from, to, SIMD_SUFFIX)
#define SIMD_CVT_2(type0, type1)					\
  SIMD_CVT(type0, type1)						\
  SIMD_CVT(type1, type0)

#if defined(AVX)
#  if defined(AVX2)
    SIMD_CVT_2(int32_t, float)		// int   <-> float

    template <> inline F64vec		// int    -> double
    cvt<double, 0>(Is32vec x)
    {
	return _mm256_cvtepi32_pd(_mm256_castsi256_si128(x));
    }

    template <> inline F64vec		// int    -> double
    cvt<double, 1>(Is32vec x)
    {
	return _mm256_cvtepi32_pd(_mm256_extractf128_si256(x, 0x1));
    }

    template <> inline Is32vec		// double -> int
    cvt<int32_t>(F64vec x, F64vec y)
    {
	return _mm256_insertf128_si256(_mm256_castsi128_si256(
					   _mm256_cvtpd_epi32(x)),
				       _mm256_cvtpd_epi32(y), 0x1);
    }

#    define SIMD_CVTI_F(itype, ftype)					\
      template <> inline vec<ftype>					\
      cvt<ftype>(vec<itype> x)						\
      {									\
	  return cvt<ftype>(cvt<int32_t>(x));				\
      }
#  else		// AVX && !AVX2
    template <> inline F32vec		// 2*int  -> float
    cvt<float>(Is32vec x, Is32vec y)
    {
	return _mm256_cvtepi32_ps(
		   _mm256_insertf128_si256(_mm256_castsi128_si256(x), y, 0x1));
    }

    template <> inline Is32vec		// float  -> int
    cvt<int32_t, 0>(F32vec x)
    {
	return _mm256_castsi256_si128(_mm256_cvtps_epi32(x));
    }

    template <> inline Is32vec		// float  -> int
    cvt<int32_t, 1>(F32vec x)
    {
	return _mm256_extractf128_si256(_mm256_cvtps_epi32(x), 0x1);
    }

    SIMD_CVT(int32_t, double)		// int    -> double

    template <> inline Is32vec		// double -> int
    cvt<int32_t>(F64vec x)
    {
	return _mm256_cvtpd_epi32(x);
    }

    template <> inline Is16vec		// float  -> short
    cvt<int16_t>(F32vec x)
    {
	__m256i	y = _mm256_cvtps_epi32(x);
	return cvt<int16_t>(vec<int32_t>(_mm256_castsi256_si128(y)),
			    vec<int32_t>(_mm256_extractf128_si256(y, 0x1)));
    }

#    define SIMD_CVTI_F(itype, ftype)					\
      template <> inline vec<ftype>					\
      cvt<ftype>(vec<itype> x)						\
      {									\
	  return SIMD_MNEMONIC(cvt, _mm256_, epi32, SIMD_SUFFIX(ftype))	\
	      (_mm256_insertf128_si256(					\
		  _mm256_castsi128_si256(cvt<int32_t>(x)),		\
		  cvt<int32_t>(shift_r<4>(x)), 0x1));			\
      }
#  endif

  SIMD_CVTI_F(int8_t,    float)		// s_char  -> float
  SIMD_CVTI_F(int16_t,   float)		// short   -> float
  SIMD_CVTI_F(u_int8_t,  float)		// u_char  -> float
  SIMD_CVTI_F(u_int16_t, float)		// u_short -> float
#  undef SIMD_CVTI_F

#elif defined(SSE2)	// !AVX && SSE2
  SIMD_CVT_2(int32_t, float)		// int    <-> float

  SIMD_CVT(int32_t, double)		// int	   -> double

  template <> inline F64vec		// int	   -> double
  cvt<double, 1>(Is32vec x)
  {
      return _mm_cvtepi32_pd(_mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2)));
  }

  template <> inline Is32vec		// double  -> int
  cvt<int32_t>(F64vec x, F64vec y)
  {
      return _mm_unpacklo_epi64(_mm_cvtpd_epi32(x), _mm_cvtpd_epi32(y));
  }

#  define SIMD_CVTI_F(itype, suffix)					\
    template <> inline F32vec						\
    cvt<float>(vec<itype> x)						\
    {									\
	return SIMD_MNEMONIC(cvt, _mm_, suffix, ps)			\
	    (_mm_movepi64_pi64(x));					\
    }
#  define SIMD_CVTF_I(itype, suffix)					\
    template <> inline vec<itype>					\
    cvt<itype>(F32vec x)						\
    {									\
	return _mm_movpi64_epi64(SIMD_MNEMONIC(cvt, _mm_, ps, suffix)	\
				 (x));					\
    }
#  define SIMD_CVT_2FI(itype, suffix)					\
    SIMD_CVTI_F(itype, suffix)						\
    SIMD_CVTF_I(itype, suffix)

  SIMD_CVT_2FI(int8_t,   pi8)		// s_char <-> float
  SIMD_CVT_2FI(int16_t,  pi16)		// short  <-> float
  SIMD_CVTI_F(u_int8_t,  pu8)		// u_char  -> float
  SIMD_CVTI_F(u_int16_t, pu16)		// u_short -> float

#  undef SIMD_CVTI_F
#  undef SIMD_CVTF_I
#  undef SIMD_CVT_2FI

#elif defined(SSE)	// !SSE2 && SSE
  template <> inline F32vec
  cvt<float>(Is32vec x, Is32vec y)	// 2*int   -> float
  {
      return _mm_cvtpi32x2_ps(x, y);
  }

  SIMD_CVT(float, int32_t)		// float   -> int

  template <> inline Is32vec		// float   -> int
  cvt<int32_t, 1>(F32vec x)
  {
      return _mm_cvtps_pi32(_mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2)));
  }

  SIMD_CVT_2(int8_t,  float)		// s_char <-> float
  SIMD_CVT_2(int16_t, float)		// short  <-> float
  SIMD_CVT(u_int8_t,  float)		// u_char  -> float
  SIMD_CVT(u_int16_t, float)		// u_short -> float
#endif
  
// [3] 浮動小数点数ベクトル間の変換
#if defined(AVX)
  template <> F64vec
  cvt<double, 0>(F32vec x)		// float -> double
  {
      return _mm256_cvtps_pd(_mm256_castps256_ps128(x));
  }
  template <> F64vec
  cvt<double, 1>(F32vec x)		// float -> double
  {
      return _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
  }

  template <> F32vec			// double -> float
  cvt<float>(F64vec x, F64vec y)
  {
      return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(x)),
				  _mm256_cvtpd_ps(y), 1);
  }
#elif defined(SSE2)
  template <> F64vec
  cvt<double, 0>(F32vec x)		// float -> double
  {
      return _mm_cvtps_pd(x);
  }
  template <> F64vec
  cvt<double, 1>(F32vec x)		// float -> double
  {
      return _mm_cvtps_pd(_mm_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2)));
  }
	  
  template <> F32vec			// double -> float
  cvt<float>(F64vec x, F64vec y)
  {
      return _mm_shuffle_ps(_mm_cvtpd_ps(x), _mm_cvtpd_ps(y),
			    _MM_SHUFFLE(1, 0, 1, 0));
  }
#endif
  
#undef SIMD_CVT
#undef SIMD_CVT_2

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_CVT_H
