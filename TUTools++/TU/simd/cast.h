/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CAST_H)
#define __TU_SIMD_CAST_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Cast operators							*
************************************************************************/
//! S型の成分を持つベクトルからT型の成分を持つベクトルへのキャストを行なう．
template <class T, class S> vec<T>	cast(vec<S> x)			;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/cast.h"
#elif defined(NEON)
#  include "TU/simd/arm/cast.h"
#endif

#endif	// !__TU_SIMD_CAST_H
