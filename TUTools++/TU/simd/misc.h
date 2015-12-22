/*
 *  $Id$
 */
#if !defined(__TU_SIMD_MISC_H)
#define __TU_SIMD_MISC_H

#include "TU/iterator.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
template <class... ITERS> inline auto
make_iterator_tuple(ITERS... iters)
    -> decltype(boost::make_tuple(
		    make_multiplex_iterator<
			lcm(iterator_value<ITERS>::size...)/
			iterator_value<ITERS>::size>(iters)...))
{
    return boost::make_tuple(
	make_multiplex_iterator<lcm(iterator_value<ITERS>::size...)/
				iterator_value<ITERS>::size>(iters)...);
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/shuffle.h"
#  include "TU/simd/intel/svml.h"
#endif

#endif	// !__TU_SIMD_MISC_H
