/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_COMPARE_H)
#define __TU_SIMD_ARM_COMPARE_H

namespace TU
{
namespace simd
{
template <class T> inline vec<T>
operator !=(vec<T> x, vec<T> y)		{ return ~(x == y); }

#define SIMD_COMPARE(func, op, type)					\
    SIMD_SPECIALIZED_FUNC(						\
	vec<mask_type<type> > func(vec<type> x, vec<type> y),		\
	op, (x, y), void, type)
#define SIMD_COMPARES(type)						\
    SIMD_COMPARE(operator ==, ceqq, type)				\
    SIMD_COMPARE(operator >,  cgtq, type)				\
    SIMD_COMPARE(operator <,  cltq, type)				\
    SIMD_COMPARE(operator >=, cgeq, type)				\
    SIMD_COMPARE(operator <=, cleq, type)

SIMD_COMPARES(int8_t)
SIMD_COMPARES(int16_t)
SIMD_COMPARES(int32_t)
SIMD_COMPARES(uint8_t)
SIMD_COMPARES(uint16_t)
SIMD_COMPARES(uint32_t)
SIMD_COMPARES(float)

#undef SIMD_COMPARE
#undef SIMD_COMPARES
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_COMPARE_H
