/*
 *  $Id$
 */
#if !defined(__TU_SIMD_INTEL_LOGICAL_H)
#define __TU_SIMD_INTEL_LOGICAL_H

#include "TU/simd/intel/logical_base.h"

namespace TU
{
namespace simd
{
template <class T> inline vec<T>
operator &(vec<T> x, vec<T> y)	{ return detail::base_and(x, y); }
template <class T> inline vec<T>
operator |(vec<T> x, vec<T> y)	{ return detail::base_or(x, y); }
template <class T> inline vec<T>
operator ^(vec<T> x, vec<T> y)	{ return detail::base_xor(x, y); }
template <class T> inline vec<T>
andnot(vec<T> x, vec<T> y)	{ return detail::base_andnot(x, y); }

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_INTEL_LOGICAL_H
