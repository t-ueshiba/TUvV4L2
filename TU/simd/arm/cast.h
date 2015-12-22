/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_CAST_H)
#define __TU_SIMD_ARM_CAST_H

namespace TU
{
namespace simd
{
template <class T, class S> inline vec<T>	cast(vec<S> x)	{ return x; }
    
#define SIMD_CAST(from, to)						\
    SIMD_SPECIALIZED_FUNC(vec<to> cast<to>(vec<from> x),		\
			  reinterpretq, (x), to, from)

SIMD_CAST(int8_t, int16_t)
SIMD_CAST(int8_t, int32_t)
SIMD_CAST(int8_t, int64_t)
SIMD_CAST(int8_t, u_int8_t)
SIMD_CAST(int8_t, u_int16_t)
SIMD_CAST(int8_t, u_int32_t)
SIMD_CAST(int8_t, u_int64_t)
SIMD_CAST(int8_t, float)

SIMD_CAST(int16_t, int8_t)
SIMD_CAST(int16_t, int32_t)
SIMD_CAST(int16_t, int64_t)
SIMD_CAST(int16_t, u_int8_t)
SIMD_CAST(int16_t, u_int16_t)
SIMD_CAST(int16_t, u_int32_t)
SIMD_CAST(int16_t, u_int64_t)
SIMD_CAST(int16_t, float)

SIMD_CAST(int32_t, int8_t)
SIMD_CAST(int32_t, int16_t)
SIMD_CAST(int32_t, int64_t)
SIMD_CAST(int32_t, u_int8_t)
SIMD_CAST(int32_t, u_int16_t)
SIMD_CAST(int32_t, u_int32_t)
SIMD_CAST(int32_t, u_int64_t)
SIMD_CAST(int32_t, float)

SIMD_CAST(int64_t, int8_t)
SIMD_CAST(int64_t, int16_t)
SIMD_CAST(int64_t, int32_t)
SIMD_CAST(int64_t, u_int8_t)
SIMD_CAST(int64_t, u_int16_t)
SIMD_CAST(int64_t, u_int32_t)
SIMD_CAST(int64_t, u_int64_t)
SIMD_CAST(int64_t, float)

SIMD_CAST(u_int8_t, int8_t)
SIMD_CAST(u_int8_t, int16_t)
SIMD_CAST(u_int8_t, int32_t)
SIMD_CAST(u_int8_t, int64_t)
SIMD_CAST(u_int8_t, u_int16_t)
SIMD_CAST(u_int8_t, u_int32_t)
SIMD_CAST(u_int8_t, u_int64_t)
SIMD_CAST(u_int8_t, float)

SIMD_CAST(u_int16_t, int8_t)
SIMD_CAST(u_int16_t, int16_t)
SIMD_CAST(u_int16_t, int32_t)
SIMD_CAST(u_int16_t, int64_t)
SIMD_CAST(u_int16_t, u_int8_t)
SIMD_CAST(u_int16_t, u_int32_t)
SIMD_CAST(u_int16_t, u_int64_t)
SIMD_CAST(u_int16_t, float)

SIMD_CAST(u_int32_t, int8_t)
SIMD_CAST(u_int32_t, int16_t)
SIMD_CAST(u_int32_t, int32_t)
SIMD_CAST(u_int32_t, int64_t)
SIMD_CAST(u_int32_t, u_int8_t)
SIMD_CAST(u_int32_t, u_int16_t)
SIMD_CAST(u_int32_t, u_int64_t)
SIMD_CAST(u_int32_t, float)

SIMD_CAST(u_int64_t, int8_t)
SIMD_CAST(u_int64_t, int16_t)
SIMD_CAST(u_int64_t, int32_t)
SIMD_CAST(u_int64_t, int64_t)
SIMD_CAST(u_int64_t, u_int8_t)
SIMD_CAST(u_int64_t, u_int16_t)
SIMD_CAST(u_int64_t, u_int32_t)
SIMD_CAST(u_int64_t, float)

SIMD_CAST(float, int8_t)
SIMD_CAST(float, int16_t)
SIMD_CAST(float, int32_t)
SIMD_CAST(float, int64_t)
SIMD_CAST(float, u_int8_t)
SIMD_CAST(float, u_int16_t)
SIMD_CAST(float, u_int32_t)
SIMD_CAST(float, u_int64_t)

#undef SIMD_CAST

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_CAST_H
