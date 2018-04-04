/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_DUP_H)
#define TU_SIMD_X86_DUP_H

#include "TU/simd/x86/unpack.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Duplication operators						*
************************************************************************/
template <bool HI, class T> inline vec<T>
dup(vec<T> x)
{
    return unpack<HI>(x, x);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_DUP_H
