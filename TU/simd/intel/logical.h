/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_LOGICAL_H)
#define __TU_SIMD_INTEL_LOGICAL_H

namespace TU
{
namespace simd
{
#define SIMD_LOGICALS(type)						\
    SIMD_BASE_FUNC(operator &, and,    type)				\
    SIMD_BASE_FUNC(operator |, or,     type)				\
    SIMD_BASE_FUNC(operator ^, xor,    type)				\
    SIMD_BASE_FUNC(andnot,     andnot, type)

SIMD_LOGICALS(int8_t)
SIMD_LOGICALS(int16_t)
SIMD_LOGICALS(int32_t)
SIMD_LOGICALS(int64_t)
SIMD_LOGICALS(u_int8_t)
SIMD_LOGICALS(u_int16_t)
SIMD_LOGICALS(u_int32_t)
SIMD_LOGICALS(u_int64_t)

#if defined(SSE)
  SIMD_LOGICALS(float)
#endif
#if defined(SSE2)
  SIMD_LOGICALS(double)
#endif

#undef SIMD_LOGICALS
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_LOGICAL_H
