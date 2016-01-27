/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_ARITHMETIC_H)
#define __TU_SIMD_X86_ARITHMETIC_H

#include "TU/simd/x86/unpack.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic and max/min operators					*
************************************************************************/
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

template <class T> inline vec<T>
operator -(vec<T> x)
{
    return zero<T>() - x;
}

#define SIMD_ADD_SUB(type)						\
    SIMD_BINARY_FUNC(operator +, add, type)				\
    SIMD_BINARY_FUNC(operator -, sub, type)

// 8/16bit整数は，飽和演算によって operator [+|-] を定義する．
#define SIMD_SAT_ADD_SUB(type)						\
    SIMD_BINARY_FUNC(operator +, adds, type)				\
    SIMD_BINARY_FUNC(operator -, subs, type)

#define SIMD_MIN_MAX(type)						\
    SIMD_BINARY_FUNC(min, min, type)					\
    SIMD_BINARY_FUNC(max, max, type)

// 加減算
SIMD_SAT_ADD_SUB(int8_t)
SIMD_SAT_ADD_SUB(int16_t)
SIMD_ADD_SUB(int32_t)
SIMD_ADD_SUB(int64_t)
SIMD_SAT_ADD_SUB(uint8_t)
SIMD_SAT_ADD_SUB(uint16_t)

// 乗算
SIMD_BINARY_FUNC(operator *, mullo, int16_t)
SIMD_BINARY_FUNC(mulhi,      mulhi, int16_t)

#if defined(SSE)
  // 加減算
  SIMD_ADD_SUB(float)

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
  SIMD_ADD_SUB(double)

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

#undef SIMD_ADD_SUB
#undef SIMD_SAT_ADD_SUB
#undef SIMD_MIN_MAX

template <class T> inline vec<T>
operator *(T c, vec<T> x)
{
    return vec<T>(c) * x;
}

template <class T> inline vec<T>
operator *(vec<T> x, T c)
{
    return x * vec<T>(c);
}

template <class T> inline vec<T>
operator /(vec<T> x, T c)
{
    return x / vec<T>(c);
}

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
template <class T> inline vec<T>
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
diff(vec<T> x, vec<T> y)	{return select(x > y, x - y, y - x);}
template <> inline Iu8vec
diff(Iu8vec x, Iu8vec y)	{return (x - y) | (y - x);}
template <> inline Iu16vec
diff(Iu16vec x, Iu16vec y)	{return (x - y) | (y - x);}
  
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_ARITHMETIC_H
