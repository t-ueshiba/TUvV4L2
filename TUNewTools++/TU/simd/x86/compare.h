/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_COMPARE_H)
#define __TU_SIMD_X86_COMPARE_H

namespace TU
{
namespace simd
{
// MMX, SSE, AVX2 には整数に対する cmplt ("less than") がない！
#define SIMD_COMPARE(func, op, type)					\
    SIMD_SPECIALIZED_FUNC(						\
	vec<mask_type<type> > func(vec<type> x, vec<type> y),		\
	op, (x, y), void, type, SIMD_SIGNED)
#define SIMD_COMPARE_R(func, op, type)					\
    SIMD_SPECIALIZED_FUNC(						\
	vec<mask_type<type> > func(vec<type> x, vec<type> y),		\
	op, (y, x), void, type, SIMD_SIGNED)
#define SIMD_COMPARES(type)						\
    SIMD_COMPARE(  operator ==, cmpeq, type)				\
    SIMD_COMPARE(  operator >,  cmpgt, type)				\
    SIMD_COMPARE_R(operator <,  cmpgt, type)

// 符号なし数に対しては等値性チェックしかできない！
SIMD_COMPARE(operator ==, cmpeq, uint8_t)
SIMD_COMPARE(operator ==, cmpeq, uint16_t)
SIMD_COMPARE(operator ==, cmpeq, uint32_t)
#if defined(SSE4)
  SIMD_COMPARE(operator ==, cmpeq, uint64_t)
#endif

SIMD_COMPARES(int8_t)
SIMD_COMPARES(int16_t)
SIMD_COMPARES(int32_t)

#if defined(AVX)	// AVX の浮動小数点数比較演算子はパラメータ形式
#  define SIMD_COMPARE_F(func, type, opcode)				\
    SIMD_SPECIALIZED_FUNC(						\
	vec<mask_type<type> > func(vec<type> x, vec<type> y),		\
	cmp, (x, y, opcode), void, type, SIMD_SUFFIX)
#  define SIMD_COMPARES_F(type)						\
    SIMD_COMPARE_F(operator ==, type, _CMP_EQ_OQ)			\
    SIMD_COMPARE_F(operator >,  type, _CMP_GT_OS)			\
    SIMD_COMPARE_F(operator <,  type, _CMP_LT_OS)			\
    SIMD_COMPARE_F(operator !=, type, _CMP_NEQ_OQ)			\
    SIMD_COMPARE_F(operator >=, type, _CMP_GE_OS)			\
    SIMD_COMPARE_F(operator <=, type, _CMP_LE_OS)

  SIMD_COMPARES_F(float)
  SIMD_COMPARES_F(double)

#  undef SIMD_COMPARE_F
#  undef SIMD_COMPARES_F
#elif defined(SSE)
#  define SIMD_COMPARES_SUP(type)					\
    SIMD_COMPARE(  operator !=, cmpneq, type)				\
    SIMD_COMPARE(  operator >=, cmpge,  type)				\
    SIMD_COMPARE_R(operator <=, cmpge,  type)

  SIMD_COMPARES(float)
  SIMD_COMPARES_SUP(float)
#  if defined(SSE2)
    SIMD_COMPARES(double)
    SIMD_COMPARES_SUP(double)
#  endif

#  undef SIMD_COMPARES_SUP
#endif

#undef SIMD_COMPARE
#undef SIMD_COMPARES
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_COMPARE_H
