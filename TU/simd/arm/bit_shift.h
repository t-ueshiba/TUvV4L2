/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_BIT_SHIFT_H)
#define __TU_SIMD_ARM_BIT_SHIFT_H

namespace TU
{
namespace simd
{
#define SIMD_SHIFT(type)						\
    SIMD_SPECIALIZED_FUNC(vec<type>  operator <<(vec<type>  x, int n),	\
			  shlq_n, (x, n), void, type)			\
    SIMD_SPECIALIZED_FUNC(vec<type>  operator >>(vec<type>  x, int n),	\
			  shrq_n, (x, n), void, type)
    
SIMD_SHIFT(int8_t)
SIMD_SHIFT(int16_t)
SIMD_SHIFT(int32_t)
SIMD_SHIFT(int64_t)
SIMD_SHIFT(u_int8_t)
SIMD_SHIFT(u_int16_t)
SIMD_SHIFT(u_int32_t)
SIMD_SHIFT(u_int64_t)

#undef SIMD_SHIFT

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_BIT_SHIFT_H
