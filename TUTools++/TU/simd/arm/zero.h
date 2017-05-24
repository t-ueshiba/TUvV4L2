/*
 *  $Id$
 */
#if !defined(TU_SIMD_ARM_ZERO_H)
#define TU_SIMD_ARM_ZERO_H

namespace TU
{
namespace simd
{
template <class T> inline vec<T>	zero()		{ return vec<T>(0); }
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ARM_ZERO_H
