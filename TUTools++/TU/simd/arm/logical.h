/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_LOGICAL_H)
#define __TU_SIMD_ARM_LOGICAL_H

namespace TU
{
namespace simd
{
template <class T> inline vec<T>
andnot(vec<T> x, vec<T> y)
{
    return (~x) & y;
}
    
#define SIMD_NOT(type)							\
    SIMD_UNARY_FUNC(operator ~, mvn, type)

#define SIMD_LOGICALS(type)						\
    SIMD_BINARY_FUNC(operator &, and, type)				\
    SIMD_BINARY_FUNC(operator |, orr, type)				\
    SIMD_BINARY_FUNC(operator ^, eor, type)

SIMD_NOT(int8_t)
SIMD_NOT(int16_t)
SIMD_NOT(int32_t)
SIMD_NOT(u_int8_t)
SIMD_NOT(u_int16_t)
SIMD_NOT(u_int32_t)

SIMD_LOGICALS(int8_t)
SIMD_LOGICALS(int16_t)
SIMD_LOGICALS(int32_t)
SIMD_LOGICALS(int64_t)
SIMD_LOGICALS(u_int8_t)
SIMD_LOGICALS(u_int16_t)
SIMD_LOGICALS(u_int32_t)
SIMD_LOGICALS(u_int64_t)

#undef SIMD_NOT
#undef SIMD_LOGICALS

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_LOGICAL_H
