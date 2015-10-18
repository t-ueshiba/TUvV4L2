/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOOKUP_H)
#define __TU_SIMD_LOOKUP_H

#include "TU/simd/insert_extract.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Lookup								*
************************************************************************/
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/lookup.h"
#elif defined(NEON)
#  include "TU/simd/arm/lookup.h"
#endif

#endif	// !__TU_SIMD_LOOKUP_H
