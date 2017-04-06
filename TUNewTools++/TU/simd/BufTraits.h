/*
 *  $Id$
 */
#if !defined(__TU_SIMD_BUFTRAITS_H)
#define __TU_SIMD_BUFTRAITS_H

#include <algorithm>
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"
#include "TU//simd/zero.h"

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
    static auto	null()
		{
		    return nullptr;
		}

    template <class IN_, class OUT_>
    static OUT_	copy(IN_ ib, IN_ ie, OUT_ out)
		{
		  // MacOS Xでは simd::store_iterator に対して
		  // std::copy() を適用しても働かない
		    for (; ib != ie; ++ib, ++out)
			*out = *ib;
		    return out;
		}

    template <class T_>
    static void	fill(iterator ib, iterator ie, const T_& c)
		{
		  // MacOS Xでは simd::store_iterator に対して
		  // std::fill() を適用しても働かない
		    for (; ib != ie; ++ib)
			*ib = c;
		}

    static void	init(iterator ib, iterator ie)
		{
		  // MacOS Xでは simd::store_iterator に対して
		  // std::fill() を適用しても働かない
		    for (; ib != ie; ++ib)
			*ib = simd::zero<T>();
		}
};
  /*
template <class... T, class ALLOC>
struct BufTraits<std::tuple<simd::vec<T>...>, ALLOC>
{
    using allocator_traits	= std::allocator_traits<simd::allocator<
					  std::tuple<simd::vec<T>...> > >;
    using iterator		= zip_iterator<
				      std::tuple<
					  simd::store_iterator<T*, true>...> >;
    using const_iterator	= zip_iterator<
				      std::tuple<
					  simd::load_iterator<T*, true>...> >;
};
  */
}	// namespace TU
#endif	// !__TU_SIMD_BUFTRAITS_H
