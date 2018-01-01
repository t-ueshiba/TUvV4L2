/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_ARITHMETIC_H)
#define TU_SIMD_X86_ARITHMETIC_H

#include "TU/simd/x86/unpack.h"
#include "TU/simd/zero.h"
#include "TU/simd/insert_extract.h"
#include "TU/simd/select.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic and max/min operators					*
************************************************************************/
template <class T> inline vec<T>
operator -(vec<T> x)
{
    return zero<T>() - x;
}

template <class T> inline vec<T>
min(vec<T> x, vec<T> y)
{
    return select(x < y, x, y);
}

template <class T> inline vec<T>
max(vec<T> x, vec<T> y)
{
    return select(x > y, x, y);
}

#define SIMD_SAT_ADD(type)						\
    SIMD_BINARY_FUNC(operator +, adds, type)

#define SIMD_ADD(type)							\
    SIMD_BINARY_FUNC(operator +, add, type)

#define SIMD_SAT_SUB(type)						\
    SIMD_BINARY_FUNC(operator -, subs, type)

#define SIMD_SUB(type)							\
    SIMD_BINARY_FUNC(operator -, sub, type)

#define SIMD_SUBS(type)							\
    SIMD_BINARY_FUNC(subs, subs, type)

#define SIMD_MIN_MAX(type)						\
    SIMD_BINARY_FUNC(min, min, type)					\
    SIMD_BINARY_FUNC(max, max, type)

// 加算
SIMD_SAT_ADD(int8_t)
SIMD_SAT_ADD(int16_t)
SIMD_ADD(int32_t)
SIMD_ADD(int64_t)
SIMD_SAT_ADD(uint8_t)
SIMD_SAT_ADD(uint16_t)

// 減算
SIMD_SAT_SUB(int8_t)
SIMD_SAT_SUB(int16_t)
SIMD_SUB(int32_t)
SIMD_SUB(int64_t)
SIMD_SUBS(uint8_t)
SIMD_SUBS(uint16_t)

// 乗算
SIMD_BINARY_FUNC(operator *, mullo, int16_t)
SIMD_BINARY_FUNC(mulhi,      mulhi, int16_t)

#if defined(SSE)
  // 加減算
  SIMD_ADD(float)
  SIMD_SUB(float)

  // 乗除算
  SIMD_BINARY_FUNC(operator *, mul, float)
  SIMD_BINARY_FUNC(operator /, div, float)

  // Min/Max
  SIMD_MIN_MAX(uint8_t)
  SIMD_MIN_MAX(int16_t)
  SIMD_MIN_MAX(float)

  // その他
  SIMD_UNARY_FUNC(sqrt,  sqrt,  float)
  SIMD_UNARY_FUNC(rsqrt, rsqrt, float)
  SIMD_UNARY_FUNC(rcp,   rcp,   float)
#endif

#if defined(SSE2)
  // 加減算
  SIMD_ADD(double)
  SIMD_SUB(double)

  // 乗除算
  SIMD_BINARY_FUNC(operator *, mul, uint32_t)
  SIMD_BINARY_FUNC(operator *, mul, double)
  SIMD_BINARY_FUNC(operator /, div, double)

  // Min/Max
  SIMD_MIN_MAX(double)

  // その他
  SIMD_UNARY_FUNC(sqrt, sqrt, double)
#endif

#if defined(SSE4)
  // 乗算
  SIMD_BINARY_FUNC(operator *, mullo, int32_t)

  // Min/Max
  SIMD_MIN_MAX(int8_t)
  SIMD_MIN_MAX(int32_t)
  SIMD_MIN_MAX(uint16_t)
  SIMD_MIN_MAX(uint32_t)
#endif

#undef SIMD_SAT_ADD
#undef SIMD_ADD
#undef SIMD_SAT_SUB
#undef SIMD_SUB
#undef SIMD_SUBS
#undef SIMD_MIN_MAX

template <bool HI, class T> inline vec<upper_type<T> >
mul(vec<T> x, vec<T> y)
{
    return unpack<HI>(x * y, mulhi(x, y));
}
    
/************************************************************************
*  Average values							*
************************************************************************/
template <class T> inline vec<T>
avg(vec<T> x, vec<T> y)			{return (x + y) >> 1;}
template <class T> inline vec<signed_type<T> >
sub_avg(vec<T> x, vec<T> y)		{return (x - y) >> 1;}

#if defined(SSE)
  SIMD_BINARY_FUNC(avg, avg, uint8_t)
  SIMD_BINARY_FUNC(avg, avg, uint16_t)
  template <> inline F32vec
  avg(F32vec x, F32vec y)		{return (x + y) * F32vec(0.5f);}
  template <> inline F32vec
  sub_avg(F32vec x, F32vec y)		{return (x - y) * F32vec(0.5f);}
#endif

#if defined(SSE2)
  template <> inline F64vec
  avg(F64vec x, F64vec y)		{return (x + y) * F64vec(0.5);}
  template <> inline F64vec
  sub_avg(F64vec x, F64vec y)		{return (x - y) * F64vec(0.5);}
#endif
  
/************************************************************************
*  Absolute values							*
************************************************************************/
template <class T> inline vec<T>
abs(vec<T> x)				{return max(x, -x);}
#if defined(SSSE3)
  SIMD_UNARY_FUNC(abs, abs, int8_t)
  SIMD_UNARY_FUNC(abs, abs, int16_t)
  SIMD_UNARY_FUNC(abs, abs, int32_t)
#endif
  
/************************************************************************
*  Absolute differences							*
************************************************************************/
template <class T> inline vec<T>
diff(vec<T> x, vec<T> y)		{return select(x > y, x - y, y - x);}
template <> inline Iu8vec
diff(Iu8vec x, Iu8vec y)		{return subs(x, y) | subs(y, x);}
template <> inline Iu16vec
diff(Iu16vec x, Iu16vec y)		{return subs(x, y) | subs(y, x);}
  
/************************************************************************
*  Fused multiply-add							*
************************************************************************/
#if defined(AVX2)
  SIMD_TRINARY_FUNC(fma, fmadd, float)
  SIMD_TRINARY_FUNC(fma, fmadd, double)
#endif

/************************************************************************
*  Horizontal addition							*
************************************************************************/
template <class T> vec<T>	hadd(vec<T> x, vec<T> y)		;
  
#if defined(SSE3)
#  define SIMD_HADD(type)	SIMD_BINARY_FUNC(hadd, hadd, type)
#  define SIMD_EMU_HADD(type)	SIMD_BINARY_FUNC(hadd, emu_hadd, type)

#  if defined(AVX2)
  SIMD_EMU_HADD(int16_t)
  SIMD_EMU_HADD(int32_t)
#  elif defined(SSE4)
  SIMD_HADD(int16_t)
  SIMD_HADD(int32_t)
#  endif
  
#  if defined(AVX)
  SIMD_EMU_HADD(float)
  SIMD_EMU_HADD(double)
#  else
  SIMD_HADD(float)
  SIMD_HADD(double)
#  endif

#  undef SIMD_HADD
#  undef SIMD_EMU_HADD
#endif
  
template <class T> inline T
hadd_impl(vec<T> x, std::integral_constant<size_t, 1>)
{
    return extract<0>(x);
}
    
template <class T, size_t I> inline T
hadd_impl(vec<T> x, std::integral_constant<size_t, I>)
{
    return hadd_impl(hadd(x, zero<T>()),
		     std::integral_constant<size_t, (I >> 1)>());
}

template <class T> inline T
hadd(vec<T> x)
{
    return hadd_impl(x, std::integral_constant<size_t, vec<T>::size>());
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_ARITHMETIC_H
