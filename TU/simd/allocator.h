/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ALLOCATOR_H)
#define __TU_SIMD_ALLOCATOR_H

#if defined(MMX)
#  include "TU/simd/x86/allocator.h"
#elif defined(NEON)
#  include "TU/simd/arm/allocator.h"
#endif

#endif	// !__TU_SIMD_ALLOCATOR_H)


