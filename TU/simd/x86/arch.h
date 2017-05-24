/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_ARCH_H)
#define TU_SIMD_X86_ARCH_H

#include <sys/types.h>
#include <immintrin.h>

/************************************************************************
*  Emulations								*
************************************************************************/
// alignr はSSSE3以降でのみサポートされるが，至便なのでemulationバージョンを定義
#if !defined(SSSE3)
  template <size_t N> inline __m64
  emu_alignr(__m64 y, __m64 x)
  {
      return _mm_or_si64(_mm_slli_si64(y, 8*(8 - N)), _mm_srli_si64(x, 8*N));
  }
#  if defined(SSE2)
  template <size_t N> inline __m128i
  emu_alignr(__m128i y, __m128i x)
  {
      return _mm_or_si128(_mm_slli_si128(y, 16 - N), _mm_srli_si128(x, N));
  }
#  endif
#endif

// AVX以降では alignr が上下のlaneに分断されて使いにくいので，自然なバージョンを定義
#if defined(AVX)
#  if defined(AVX2)
  template <size_t N> inline __m256i
  emu_alignr_impl(__m256i y, __m256i x, std::true_type)
  {
      return _mm256_alignr_epi8(_mm256_permute2f128_si256(x, y, 0x21), x, N);
  }
  template <size_t N> inline __m256i
  emu_alignr_impl(__m256i y, __m256i x, std::false_type)
  {
      return _mm256_alignr_epi8(y, _mm256_permute2f128_si256(x, y, 0x21),
				N - 16);
  }
#  else
  template <size_t N> inline __m256i
  emu_alignr_impl(__m256i y, __m256i x, std::true_type)
  {
      return _mm256_insertf128_si256(
		 _mm256_insertf128_si256(
		     _mm256_setzero_si256(),
		     _mm_alignr_epi8(_mm256_extractf128_si256(x, 0x1),
				     _mm256_extractf128_si256(x, 0x0),
				     N),
		     0x0),
		 _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x0),
				 _mm256_extractf128_si256(x, 0x1),
				 N),
		 0x1);
  }
  template <size_t N> inline __m256i
  emu_alignr_impl(__m256i y, __m256i x, std::false_type)
  {
      return _mm256_insertf128_si256(
		 _mm256_insertf128_si256(
		     _mm256_setzero_si256(),
		     _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x0),
				     _mm256_extractf128_si256(x, 0x1),
				     N - 16),
		     0x0),
		 _mm_alignr_epi8(_mm256_extractf128_si256(y, 0x1),
				 _mm256_extractf128_si256(y, 0x0),
				 N - 16),
		 0x1);
  }
#  endif
  template <size_t N> inline __m256i
  emu_alignr(__m256i y, __m256i x)
  {
      return emu_alignr_impl<N>(y, x, std::integral_constant<bool, (N < 16)>());
  }
#endif

// AVX以降では srli_si256, slli_si256 が上下のlaneに分断されて使いにくいので，
// 自然なバージョンを定義
#if defined(AVX)
#  if defined(AVX2)
  template <size_t N> inline __m256i
  emu_srli(__m256i x)
  {
      return _mm256_alignr_epi8(_mm256_permute2f128_si256(x, x, 0x81), x, N);
  }

  template <size_t N> inline __m256i
  emu_slli_impl(__m256i x, std::true_type)
  {
      return _mm256_alignr_epi8(x, _mm256_permute2f128_si256(x, x, 0x08),
				16 - N);
  }
  template <size_t N> inline __m256i
  emu_slli_impl(__m256i x, std::false_type)
  {
      return _mm256_alignr_epi8(_mm256_permute2f128_si256(x, x, 0x08),
				_mm256_setzero_si256(), 32 - N);
  }
#  else
  template <size_t N> inline __m256i
  emu_srli(__m256i x)
  {
      __m128i	y = _mm256_extractf128_si256(x, 0x1);
      return _mm256_insertf128_si256(
		 _mm256_insertf128_si256(
		     _mm256_setzero_si256(),
		     _mm_alignr_epi8(y, _mm256_extractf128_si256(x, 0x0), N),
		     0x0),
		 _mm_srli_si128(y, N),
		 0x1);
  }

  template <size_t N> inline __m256i
  emu_slli_impl(__m256i x, std::true_type)
  {
      __m128i	y = _mm256_extractf128_si256(x, 0x0);
      return _mm256_insertf128_si256(
		 _mm256_insertf128_si256(_mm256_setzero_si256(),
					 _mm_slli_si128(y, N),
					 0x0),
		 _mm_alignr_epi8(_mm256_extractf128_si256(x, 0x1), y, 16 - N),
		 0x1);
  }
  template <size_t N> inline __m256i
  emu_slli_impl(__m256i x, std::false_type)
  {
      __m128i	y = _mm256_extractf128_si256(x, 0x0);

      return _mm256_insertf128_si256(
		 _mm256_setzero_si256(),
		 _mm_alignr_epi8(y, _mm_setzero_si128(), 32 - N),
		 0x1);
  }
#  endif
  template <size_t N> inline __m256i
  emu_slli(__m256i x)
  {
      return emu_slli_impl<N>(x, std::integral_constant<bool, (N < 16)>());
  }
#endif

/************************************************************************
*  Macros for constructing mnemonics of intrinsics			*
************************************************************************/
#define SIMD_PREFIX(type)	SIMD_PREFIX_##type
#define SIMD_SUFFIX(type)	SIMD_SUFFIX_##type
#define SIMD_SIGNED(type)	SIMD_SIGNED_##type
#define SIMD_BASE(type)		SIMD_BASE_##type

#if defined(AVX2)
#  define SIMD_PREFIX_int8_t	_mm256_
#  define SIMD_PREFIX_int16_t	_mm256_
#  define SIMD_PREFIX_int32_t	_mm256_
#  define SIMD_PREFIX_int64_t	_mm256_
#  define SIMD_PREFIX_uint8_t	_mm256_
#  define SIMD_PREFIX_uint16_t	_mm256_
#  define SIMD_PREFIX_uint32_t	_mm256_
#  define SIMD_PREFIX_uint64_t	_mm256_
#  define SIMD_PREFIX_ivec_t	_mm256_
#else
#  define SIMD_PREFIX_int8_t	_mm_
#  define SIMD_PREFIX_int16_t	_mm_
#  define SIMD_PREFIX_int32_t	_mm_
#  define SIMD_PREFIX_int64_t	_mm_
#  define SIMD_PREFIX_uint8_t	_mm_
#  define SIMD_PREFIX_uint16_t	_mm_
#  define SIMD_PREFIX_uint32_t	_mm_
#  define SIMD_PREFIX_uint64_t	_mm_
#  define SIMD_PREFIX_ivec_t	_mm_
#endif

#if defined(SSE2)
#  define SIMD_SUFFIX_int8_t	epi8
#  define SIMD_SUFFIX_int16_t	epi16
#  define SIMD_SUFFIX_int32_t	epi32
#  define SIMD_SUFFIX_int64_t	epi64
#  define SIMD_SUFFIX_uint8_t	epu8
#  define SIMD_SUFFIX_uint16_t	epu16
#  define SIMD_SUFFIX_uint32_t	epu32
#  define SIMD_SUFFIX_uint64_t	epi64
#  if defined(AVX2)
#    define SIMD_SUFFIX_ivec_t	si256
#  else
#    define SIMD_SUFFIX_ivec_t	si128
#  endif
#else
#  define SIMD_SUFFIX_int8_t	pi8
#  define SIMD_SUFFIX_int16_t	pi16
#  define SIMD_SUFFIX_int32_t	pi32
#  define SIMD_SUFFIX_int64_t	si64
#  define SIMD_SUFFIX_uint8_t	pu8
#  define SIMD_SUFFIX_uint16_t	pu16
#  define SIMD_SUFFIX_uint32_t	pu32
#  define SIMD_SUFFIX_uint64_t	si64
#  define SIMD_SUFFIX_ivec_t	si64
#endif
#define SIMD_SUFFIX_void

#define SIMD_SIGNED_int8_t	SIMD_SUFFIX_int8_t
#define SIMD_SIGNED_int16_t	SIMD_SUFFIX_int16_t
#define SIMD_SIGNED_int32_t	SIMD_SUFFIX_int32_t
#define SIMD_SIGNED_int64_t	SIMD_SUFFIX_int64_t
#define SIMD_SIGNED_uint8_t	SIMD_SUFFIX_int8_t
#define SIMD_SIGNED_uint16_t	SIMD_SUFFIX_int16_t
#define SIMD_SIGNED_uint32_t	SIMD_SUFFIX_int32_t
#define SIMD_SIGNED_uint64_t	SIMD_SUFFIX_int64_t
    
#define SIMD_BASE_int8_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_int16_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_int32_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_int64_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_uint8_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_uint16_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_uint32_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_uint64_t	SIMD_SUFFIX_ivec_t
#define SIMD_BASE_ivec_t	SIMD_SUFFIX_ivec_t

#if defined(AVX)
#  define SIMD_PREFIX_float	_mm256_
#  define SIMD_PREFIX_fvec_t	_mm256_
#  define SIMD_PREFIX_double	_mm256_
#  define SIMD_PREFIX_dvec_t	_mm256_
#elif defined(SSE)
#  define SIMD_PREFIX_float	_mm_
#  define SIMD_PREFIX_fvec_t	_mm_
#  if defined(SSE2)
#    define SIMD_PREFIX_double	_mm_
#    define SIMD_PREFIX_dvec_t	_mm_
#  endif
#endif

#if defined(SSE)
#  define SIMD_SUFFIX_float	ps
#  define SIMD_SUFFIX_fvec_t	ps
#  define SIMD_SIGNED_float	ps
#  define SIMD_BASE_float	ps
#  define SIMD_BASE_fvec_t	ps
#endif    
#if defined(SSE2)
#  define SIMD_SUFFIX_double	pd
#  define SIMD_SUFFIX_dvec_t	pd
#  define SIMD_SIGNED_double	pd
#  define SIMD_BASE_double	pd
#  define SIMD_BASE_dvec_t	pd
#endif    

#define SIMD_CAT(op, prefix, from, suffix)	prefix##op##from##_##suffix
#define SIMD_MNEMONIC(op, prefix, from, suffix)				\
    SIMD_CAT(op, prefix, from, suffix)

#define SIMD_FUNC(signature, op, args, from, to, suffix)		\
    inline signature							\
    {									\
	return SIMD_MNEMONIC(op, SIMD_PREFIX(to),			\
			     SIMD_SUFFIX(from), suffix(to))args;	\
    }

#define SIMD_SPECIALIZED_FUNC(signature, op, args, from, to, suffix)	\
    template <> SIMD_FUNC(signature, op, args, from, to, suffix)

#define SIMD_SIGNED_FUNC(func, op, type)				\
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x, vec<type> y),	\
			  op, (x, y), void, type, SIMD_SIGNED)

#define SIMD_UNARY_FUNC(func, op, type)					\
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x),			\
			  op, (x), void, type, SIMD_SUFFIX)

#define SIMD_BINARY_FUNC(func, op, type)				\
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x, vec<type> y),	\
			  op, (x, y), void, type, SIMD_SUFFIX)

#endif	// !TU_SIMD_X86_ARCH_H
