/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ZERO_H)
#define __TU_SIMD_ZERO_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Zero-vector generators						*
************************************************************************/
//! 全成分が0であるベクトルを生成する．
template <class T> vec<T>	zero()					;
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/zero.h"
#elif defined(NEON)
#  include "TU/simd/arm/zero.h"
#endif

#endif	// !__TU_SIMD_ZERO_H
