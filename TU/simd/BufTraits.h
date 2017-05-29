/*
 *  $Id$
 */
#if !defined(TU_SIMD_BUFTRAITS_H)
#define TU_SIMD_BUFTRAITS_H

#include "TU/Array++.h"
#include "TU/simd/allocator.h"
#include "TU/simd/store_iterator.h"
#include "TU/simd/load_iterator.h"

namespace TU
{
template <class T, class ALLOC>
class BufTraits<simd::vec<T>, ALLOC>
    : public std::allocator_traits<simd::allocator<simd::vec<T> > >
{
  private:
    using super			= std::allocator_traits<
				      simd::allocator<simd::vec<T> > >;

  public:
    using iterator		= simd::store_iterator<T, true>;
    using const_iterator	= simd::load_iterator<T, true>;
    
  protected:
    using			typename super::pointer;

    constexpr static size_t	Alignment = sizeof(simd::vec<T>);
    
    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
};

template <class T>
class BufTraits<T, simd::allocator<T> >
    : public std::allocator_traits<simd::allocator<T> >
{
  private:
    using super			= std::allocator_traits<simd::allocator<T > >;

  public:
    using iterator		= simd::ptr<T>;
    using const_iterator	= simd::ptr<const T>;
    
  protected:
    using			typename super::pointer;

    constexpr static size_t	Alignment = sizeof(simd::vec<T>);
    
    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
};

}	// namespace TU
#endif	// !TU_SIMD_BUFTRAITS_H
