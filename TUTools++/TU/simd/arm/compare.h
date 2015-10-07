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
    template <> vec<type>						\
    func(vec<type> x, vec<type> y)					\
    {									\
	return								\
	    cast<type>(SIMD_MNEMONIC(op, , SIMD_SUFFIX(type))(x, y));	\
    }
#define SIMD_COMPARES(type)						\
    SIMD_COMPARE(operator ==, ceqq, type)				\
    SIMD_COMPARE(operator >,  cgtq, type)				\
    SIMD_COMPARE(operator <,  cltq, type)				\
    SIMD_COMPARE(operator >=, cgeq, type)				\
    SIMD_COMPARE(operator <=, cleq, type)

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
