/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_ARITHMETIC_H)
#define __TU_SIMD_ARM_ARITHMETIC_H

namespace TU
{
namespace simd
{
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
#define SIMD_ADD_SUB(type)						\
    SIMD_BINARY_FUNC(operator +, addq, type)				\
    SIMD_BINARY_FUNC(operator -, subq, type)

// 8/16bit整数は，飽和演算によって operator [+|-] を定義する．
#define SIMD_SAT_ADD_SUB(type)						\
    SIMD_BINARY_FUNC(operator +, qaddq, type)				\
    SIMD_BINARY_FUNC(operator -, qsubq, type)

#define SIMD_MUL(type)							\
    SIMD_BINARY_FUNC(operator *, mulq, type)

#define SIMD_NEGATE(type)						\
    SIMD_UNARY_FUNC(operator -, negq, type)

#define SIMD_MIN_MAX(type)						\
    SIMD_BINARY_FUNC(min, minq, type)					\
    SIMD_BINARY_FUNC(max, maxq, type)

#define SIMD_RCP_RSQRT(type)						\
    SIMD_UNARY_FUNC(rcp, recpeq, type)					\
    SIMD_UNARY_FUNC(rsqrt, rsqrteq, type)

  // 加減算
SIMD_SAT_ADD_SUB(int8_t)
SIMD_SAT_ADD_SUB(int16_t)
SIMD_ADD_SUB(int32_t)
SIMD_ADD_SUB(int64_t)
SIMD_SAT_ADD_SUB(u_int8_t)
SIMD_SAT_ADD_SUB(u_int16_t)
SIMD_ADD_SUB(u_int32_t)
SIMD_ADD_SUB(u_int64_t)
SIMD_ADD_SUB(float)

// 乗算
SIMD_MUL(int8_t)
SIMD_MUL(int16_t)
SIMD_MUL(int32_t)
SIMD_MUL(u_int8_t)
SIMD_MUL(u_int16_t)
SIMD_MUL(u_int32_t)
SIMD_MUL(float)

// 符号反転
SIMD_NEGATE(int8_t)
SIMD_NEGATE(int16_t)
SIMD_NEGATE(int32_t)
SIMD_NEGATE(float)

// min/max
SIMD_MIN_MAX(int8_t)
SIMD_MIN_MAX(int16_t)
SIMD_MIN_MAX(int32_t)
SIMD_MIN_MAX(u_int8_t)
SIMD_MIN_MAX(u_int16_t)
SIMD_MIN_MAX(u_int32_t)
SIMD_MIN_MAX(float)

SIMD_RCP_RSQRT(u_int32_t)
SIMD_RCP_RSQRT(float)

#undef SIMD_ADD_SUB
#undef SIMD_SAT_ADD_SUB
#undef SIMD_MUL
#undef SIMD_NEGATE
#undef SIMD_MIN_MAX
#undef SIMD_RCP_RSQRT
  
/************************************************************************
*  Average values							*
************************************************************************/
#define SIMD_AVG_SUB_AVG(type)						\
    SIMD_BINARY_FUNC(avg, haddq, type)					\
    SIMD_BINARY_FUNC(sub_avg, hsubq, type)

SIMD_AVG_SUB_AVG(int8_t)
SIMD_AVG_SUB_AVG(int16_t)
SIMD_AVG_SUB_AVG(int32_t)
SIMD_AVG_SUB_AVG(u_int8_t)
SIMD_AVG_SUB_AVG(u_int16_t)
SIMD_AVG_SUB_AVG(u_int32_t)

template <> inline F32vec
avg(F32vec x, F32vec y)			{return (x + y) * F32vec(0.5f);}

template <> inline F32vec
sub_avg(F32vec x, F32vec y)		{return (x - y) * F32vec(0.5f);}
    
/************************************************************************
*  Absolute values							*
************************************************************************/
#define SIMD_ABS(type)							\
    SIMD_UNARY_FUNC(abs, absq, type)

SIMD_ABS(int8_t)
SIMD_ABS(int16_t)
SIMD_ABS(int32_t)
SIMD_ABS(float)

#undef SIMD_ABS

/************************************************************************
*  Absolute differences							*
************************************************************************/
#define SIMD_DIFF(type)							\
    SIMD_BINARY_FUNC(diff, abdq, type)

SIMD_DIFF(int8_t)
SIMD_DIFF(int16_t)
SIMD_DIFF(int32_t)
SIMD_DIFF(u_int8_t)
SIMD_DIFF(u_int16_t)
SIMD_DIFF(u_int32_t)
SIMD_DIFF(float)

#undef SIMD_DIFF

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_ARITHMETIC_H
