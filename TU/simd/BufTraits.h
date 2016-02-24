/*
 *  $Id$
 */
#if !defined(__TU_SIMD_BUFTRAITS_H)
#define __TU_SIMD_BUFTRAITS_H

#include <algorithm>
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"

namespace TU
{
template <class T, class ALLOC>	struct BufTraits;
template <class T, class ALLOC>
struct BufTraits<simd::vec<T>, ALLOC>
{
    typedef simd::allocator<simd::vec<T> >	allocator_type;
    typedef simd::store_iterator<T*, true>	iterator;
    typedef simd::load_iterator<const T*, true>	const_iterator;

    template <class IN_, class OUT_>
    static OUT_	copy(IN_ ib, IN_ ie, OUT_ out)
		{
		    return std::copy(ib, ie, out);
		}

    template <class ITER_, class T_>
    static void	fill(ITER_ ib, ITER_ ie, const T_& c)
		{
		    std::fill(ib, ie, c);
		}
};
    
}	// namespace TU
#endif	// !__TU_SIMD_BUFTRAITS_H
