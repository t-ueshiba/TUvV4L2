/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_ZERO_H)
#define __TU_SIMD_ARM_ZERO_H

namespace TU
{
namespace simd
{
template <class T> inline vec<T>	zero()		{ return vec<T>(0); }
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_ZERO_H
