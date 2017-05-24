/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_SHIFT_H)
#define TU_SIMD_X86_SHIFT_H

namespace TU
{
namespace simd
{
#if defined(AVX2) || !defined(SSSE3)
#  define SIMD_ELM_SHIFT_R_I2(type)					\
    template <size_t N>	inline vec<type>				\
    shift_r(vec<type> x, vec<type> y)					\
    {									\
	return emu_alignr<N*vec<type>::element_size>(y, x);		\
    }
#else
#  define SIMD_ELM_SHIFT_R_I2(type)					\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_r(vec<type> x, vec<type> y),		\
	      alignr, (y, x, N*vec<type>::element_size),		\
	      void, int8_t, SIMD_SIGNED)
#endif
SIMD_ELM_SHIFT_R_I2(int8_t)
SIMD_ELM_SHIFT_R_I2(int16_t)
SIMD_ELM_SHIFT_R_I2(int32_t)
SIMD_ELM_SHIFT_R_I2(int64_t)
SIMD_ELM_SHIFT_R_I2(uint8_t)
SIMD_ELM_SHIFT_R_I2(uint16_t)
SIMD_ELM_SHIFT_R_I2(uint32_t)
SIMD_ELM_SHIFT_R_I2(uint64_t)

#undef MM_ELM_SHIFT_R_I2

// 整数ベクトルの要素シフト（実装上の注意：MMXでは64bit整数のシフトは
// bit単位だが，SSE2以上の128bit整数ではbyte単位である．また，AVX2では
// 上下のlaneに分断されないemulationバージョンを使う．）
#if defined(AVX2)
#  define SIMD_ELM_SHIFTS_I(type)					\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N> vec<type>					\
    shift_l(vec<type> x)						\
    {									\
	return emu_slli<N*vec<type>::element_size>(x);			\
    }									\
    template <size_t N> vec<type>					\
    shift_r(vec<type> x)						\
    {									\
	return emu_srli<N*vec<type>::element_size>(x);			\
    }
#elif defined(SSE2)
#  define SIMD_ELM_SHIFTS_I(type)					\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_l(vec<type> x), slli,			\
	      (x, N*vec<type>::element_size), void, type, SIMD_BASE)	\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_r(vec<type> x), srli,			\
	      (x, N*vec<type>::element_size), void, type, SIMD_BASE)
#else
#  define SIMD_ELM_SHIFTS_I(type)					\
    template <> inline vec<type>					\
    shift_l<0>(vec<type> x)				{return x;}	\
    template <> inline vec<type>					\
    shift_r<0>(vec<type> x)				{return x;}	\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_l(vec<type> x), slli,			\
	      (x, 8*N*vec<type>::element_size), void, type, SIMD_BASE)	\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_r(vec<type> x), srli,			\
	      (x, 8*N*vec<type>::element_size), void, type, SIMD_BASE)
#endif

SIMD_ELM_SHIFTS_I(int8_t)
SIMD_ELM_SHIFTS_I(int16_t)
SIMD_ELM_SHIFTS_I(int32_t)
SIMD_ELM_SHIFTS_I(int64_t)
SIMD_ELM_SHIFTS_I(uint8_t)
SIMD_ELM_SHIFTS_I(uint16_t)
SIMD_ELM_SHIFTS_I(uint32_t)
SIMD_ELM_SHIFTS_I(uint64_t)

#undef SIMD_ELM_SHIFTS_I

// 浮動小数点数ベクトルの要素シフト
#if !defined(AVX2) && defined(AVX)
  template <size_t N> inline F32vec
  shift_l(F32vec x)
  {
      return _mm256_castsi256_ps(
		 emu_slli<N*F32vec::element_size>(_mm256_castps_si256(x)));
  }

  template <size_t N> inline F32vec
  shift_r(F32vec x)
  {
      return _mm256_castsi256_ps(
		 emu_srli<N*F32vec::element_size>(_mm256_castps_si256(x)));
  }

  template <size_t N> inline F64vec
  shift_l(F64vec x)
  {
      return _mm256_castsi256_pd(
		 emu_slli<N*F64vec::element_size>(_mm256_castpd_si256(x)));
  }

  template <size_t N> inline F64vec
  shift_r(F64vec x)
  {
      return _mm256_castsi256_pd(
		 emu_srli<N*F64vec::element_size>(_mm256_castpd_si256(x)));
  }
#elif defined(SSE2)
  template <size_t N> inline F32vec
  shift_l(F32vec x)
  {
      return cast<float>(shift_l<N>(cast<uint32_t>(x)));
  }

  template <size_t N> inline F32vec
  shift_r(F32vec x)
  {
      return cast<float>(shift_r<N>(cast<uint32_t>(x)));
  }

  template <size_t N> inline F64vec
  shift_l(F64vec x)
  {
      return cast<double>(shift_l<N>(cast<uint64_t>(x)));
  }

  template <size_t N> inline F64vec
  shift_r(F64vec x)
  {
      return cast<double>(shift_r<N>(cast<uint64_t>(x)));
  }
#endif
    
}	// namespace simd
}	// namespace TU

#endif	// !TU_SIMD_X86_SHIFT_H
