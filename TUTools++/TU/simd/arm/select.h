/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_SELECT_H)
#define __TU_SIMD_ARM_SELECT_H

namespace TU
{
namespace simd
{
#define SIMD_SELECT(type)						\
    SIMD_SPECIALIZED_FUNC(						\
	vec<type> select(vec<type_traits<type>::unsigned_type> mask,	\
			 vec<type> x, vec<type> y),			\
	bslq, (mask, x, y), void, type)					\

SIMD_SELECT(int8_t)
SIMD_SELECT(int16_t)
SIMD_SELECT(int32_t)
SIMD_SELECT(int64_t)
SIMD_SELECT(u_int8_t)
SIMD_SELECT(u_int16_t)
SIMD_SELECT(u_int32_t)
SIMD_SELECT(u_int64_t)
SIMD_SELECT(float)
  
#undef SIMD_SELECT

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_SELECT_H
