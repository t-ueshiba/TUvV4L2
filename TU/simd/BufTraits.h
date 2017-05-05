/*
 *  $Id$
 */
#if !defined(__TU_SIMD_BUFTRAITS_H)
#define __TU_SIMD_BUFTRAITS_H

#include <algorithm>
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/zero.h"

namespace TU
{
template <class T, class ALLOC>	struct BufTraits;
template <class T, class ALLOC>
struct BufTraits<simd::vec<T>, ALLOC>
{
    using allocator_traits	= std::allocator_traits<
				      simd::allocator<simd::vec<T> > >;
    using iterator		= simd::store_iterator<T*, true>;
    using const_iterator	= simd::load_iterator<const T*, true>;

  protected:
    static auto	null()		{ return nullptr; }
};

}	// namespace TU
#endif	// !__TU_SIMD_BUFTRAITS_H
