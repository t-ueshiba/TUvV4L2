/*
 *  $Id$
 */
#if !defined(__TU_SIMD_DUP_H)
#define __TU_SIMD_DUP_H

#include "TU/simd/cvt.h"
#include "TU/simd/logical.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Duplication operators						*
************************************************************************/
template <bool HI, class T> vec<T>	dup(vec<T> x)			;
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/dup.h"
#elif defined(NEON)
#  include "TU/simd/arm/dup.h"
#endif

#endif	// !__TU_SIMD_DUP_H
