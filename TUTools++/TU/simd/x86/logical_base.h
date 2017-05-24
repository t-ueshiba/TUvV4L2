/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_LOGICAL_BASE_H)
#define TU_SIMD_X86_LOGICAL_BASE_H

namespace TU
{
namespace simd
{
namespace detail
{
#define SIMD_LOGICALS(type)						\
    SIMD_FUNC(type base_and(type x, type y),				\
	      and, (x, y), void, type, SIMD_BASE)			\
    SIMD_FUNC(type base_or(type x, type y),				\
	      or, (x, y), void, type, SIMD_BASE)			\
    SIMD_FUNC(type base_xor(type x, type y),				\
	      xor, (x, y), void, type, SIMD_BASE)			\
    SIMD_FUNC(type base_andnot(type x, type y),				\
	      andnot, (x, y), void, type, SIMD_BASE)

SIMD_LOGICALS(ivec_t)
#if defined(SSE)
  SIMD_LOGICALS(fvec_t)
#endif
#if defined(SSE2)
  SIMD_LOGICALS(dvec_t)
#endif

#undef SIMD_BASE_LOGICALS

}	// namespace detail
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_LOGICAL_BASE_H
