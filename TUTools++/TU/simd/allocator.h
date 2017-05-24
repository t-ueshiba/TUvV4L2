/*
 *  $Id$
 */
#if !defined(TU_SIMD_ALLOCATOR_H)
#define TU_SIMD_ALLOCATOR_H

#include "TU/simd/vec.h"

#if defined(MMX)
#  include "TU/simd/x86/allocator.h"
#elif defined(NEON)
#  include "TU/simd/arm/allocator.h"
#endif

#endif	// !TU_SIMD_ALLOCATOR_H)


