/*
 *  $Id$
 */
#if !defined(TU_SIMD_MISC_H)
#define TU_SIMD_MISC_H

#include "TU/iterator.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
template <class... ITERS> inline auto
make_iterator_tuple(ITERS... iters)
{
    return std::make_tuple(
	make_multiplex_iterator<lcm(iterator_value<ITERS>::size...)/
				iterator_value<ITERS>::size>(iters)...);
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/shuffle.h"
#  include "TU/simd/x86/svml.h"
#endif

#endif	// !TU_SIMD_MISC_H
