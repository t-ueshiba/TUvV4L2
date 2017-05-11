/*
 *  $Id$
 */
#if !defined(__TU_SIMD_BUFTRAITS_H)
#define __TU_SIMD_BUFTRAITS_H

#include <memory>
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"
#include "TU/simd/zero.h"

namespace TU
{
template <class T, class ALLOC>	class BufTraits;
template <class T, class ALLOC>
class BufTraits<simd::vec<T>, ALLOC>
    : public std::allocator_traits<simd::allocator<simd::vec<T> > >
{
  private:
    using super			= std::allocator_traits<
				      simd::allocator<simd::vec<T> > >;
  public:
    using iterator		= simd::store_iterator<T*, true>;
    using const_iterator	= simd::load_iterator<const T*, true>;
    
  protected:
    using			typename super::pointer;

    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
};

}	// namespace TU
#endif	// !__TU_SIMD_BUFTRAITS_H
