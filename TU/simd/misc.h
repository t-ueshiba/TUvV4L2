/*
 *  $Id$
 */
#if !defined(__TU_SIMD_MISC_H)
#define __TU_SIMD_MISC_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Control functions							*
************************************************************************/
void	empty()								;
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/misc.h"
#  include "TU/simd/intel/shuffle.h"
#  include "TU/simd/intel/svml.h"
#elif defined(NEON)
#  include "TU/simd/arm/misc.h"
#endif

#endif	// !__TU_SIMD_MISC_H
