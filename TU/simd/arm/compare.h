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
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x, vec<type> y),	\
			  op, (x, y), q, type)
#define SIMD_COMPARES(type)						\
    SIMD_COMPARE(operator ==, ceq, type)				\
    SIMD_COMPARE(operator >,  cgt, type)				\
    SIMD_COMPARE(operator <,  clt, type)				\
    SIMD_COMPARE(operator >=, cge, type)				\
    SIMD_COMPARE(operator <=, cle, type)

SIMD_COMPARES(int8_t)
SIMD_COMPARES(int16_t)
SIMD_COMPARES(int32_t)
SIMD_COMPARES(u_int8_t)
SIMD_COMPARES(u_int16_t)
SIMD_COMPARES(u_int32_t)
SIMD_COMPARES(float)

#undef SIMD_COMPARE
#undef SIMD_COMPARES
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_COMPARE_H
