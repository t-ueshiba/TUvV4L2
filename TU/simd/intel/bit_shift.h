/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_BIT_SHIFT_H)
#define __TU_SIMD_INTEL_BIT_SHIFT_H

namespace TU
{
namespace simd
{
#define SIMD_LOGICAL_SHIFT_LEFT(type)					\
    SIMD_SPECIALIZED_FUNC(vec<type> operator <<(vec<type> x, int n),	\
			  slli, (x, n), void, type, SIMD_SIGNED)
#define SIMD_LOGICAL_SHIFT_RIGHT(type)					\
    SIMD_SPECIALIZED_FUNC(vec<type> operator >>(vec<type> x, int n),	\
			  srli, (x, n), void, type, SIMD_SIGNED)
#define SIMD_NUMERIC_SHIFT_RIGHT(type)					\
    SIMD_SPECIALIZED_FUNC(vec<type> operator >>(vec<type> x, int n),	\
			  srai, (x, n), void, type, SIMD_SIGNED)

SIMD_LOGICAL_SHIFT_LEFT(int16_t)
SIMD_LOGICAL_SHIFT_LEFT(int32_t)
SIMD_LOGICAL_SHIFT_LEFT(int64_t)
SIMD_LOGICAL_SHIFT_LEFT(u_int16_t)
SIMD_LOGICAL_SHIFT_LEFT(u_int32_t)
SIMD_LOGICAL_SHIFT_LEFT(u_int64_t)

SIMD_NUMERIC_SHIFT_RIGHT(int16_t)
SIMD_NUMERIC_SHIFT_RIGHT(int32_t)
SIMD_LOGICAL_SHIFT_RIGHT(u_int16_t)
SIMD_LOGICAL_SHIFT_RIGHT(u_int32_t)
SIMD_LOGICAL_SHIFT_RIGHT(u_int64_t)

#undef SIMD_LOGICAL_SHIFT_LEFT
#undef SIMD_LOGICAL_SHIFT_RIGHT
#undef SIMD_NUMERIC_SHIFT_RIGHT

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_BIT_SHIFT_H
