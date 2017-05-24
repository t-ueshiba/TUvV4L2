/*
 *  $Id$
 */
#if !defined(TU_SIMD_ARM_SHIFT_H)
#define TU_SIMD_ARM_SHIFT_H

namespace TU
{
namespace simd
{
#define SIMD_SHIFT_R(type)						\
    template <size_t N>							\
    SIMD_FUNC(vec<type> shift_r(vec<type> x, vec<type> y),		\
	      extq, (x, y, N), void, type)

SIMD_SHIFT_R(int8_t)
SIMD_SHIFT_R(int16_t)
SIMD_SHIFT_R(int32_t)
SIMD_SHIFT_R(int64_t)
SIMD_SHIFT_R(uint8_t)
SIMD_SHIFT_R(uint16_t)
SIMD_SHIFT_R(uint32_t)
SIMD_SHIFT_R(uint64_t)

#undef SIMD_SHIFT_R
    
}	// namespace simd
}	// namespace TU

#endif	// !TU_SIMD_ARM_SHIFT_H
