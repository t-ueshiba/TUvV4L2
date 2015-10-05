/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_LOAD_STORE_H)
#define __TU_SIMD_ARM_LOAD_STORE_H

namespace TU
{
namespace simd
{
#define SIMD_LOAD_STORE(type)						\
    SIMD_SPECIALIZED_FUNC(vec<type> load<true>(const type* p),		\
			  ld1q, (p), void, type)			\
    SIMD_SPECIALIZED_FUNC(vec<type> load<false>(const type* p),		\
			  ld1q, (p), void, type)			\
    SIMD_SPECIALIZED_FUNC(void store<true>(type* p, vec<type> x),	\
			  st1q, (p, x), void, type)			\
    SIMD_SPECIALIZED_FUNC(void store<false>(type* p, vec<type> x),	\
			  st1q, (p, x), void, type)

SIMD_LOAD_STORE(int8_t)
SIMD_LOAD_STORE(int16_t)
SIMD_LOAD_STORE(int32_t)
SIMD_LOAD_STORE(int64_t)
SIMD_LOAD_STORE(u_int8_t)
SIMD_LOAD_STORE(u_int16_t)
SIMD_LOAD_STORE(u_int32_t)
SIMD_LOAD_STORE(u_int64_t)
SIMD_LOAD_STORE(float)

#undef SIMD_LOAD_STORE

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_LOAD_STORE_H
