/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_INSERT_EXTRACT_H)
#define __TU_SIMD_ARM_INSERT_EXTRACT_H

namespace TU
{
namespace simd
{
#define SIMD_INSERT_EXTRACT(type)					\
    template <size_t I>							\
    SIMD_FUNC(vec<type> insert(vec<type> x, type val), setq_lane,	\
	      (val, x, I), void, type)					\
    template <size_t I>							\
    SIMD_FUNC(type extract(vec<type> x), getq_lane,			\
	      (x, I), void, type)

SIMD_INSERT_EXTRACT(int8_t)
SIMD_INSERT_EXTRACT(int16_t)
SIMD_INSERT_EXTRACT(int32_t)
SIMD_INSERT_EXTRACT(uint8_t)
SIMD_INSERT_EXTRACT(uint16_t)
SIMD_INSERT_EXTRACT(uint32_t)
SIMD_INSERT_EXTRACT(float)

#undef SIMD_INSERT_EXTRACT
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_INSERT_EXTRACT_H
