/*!
  \file		BufTraits.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルに関連したバッファの特性
*/
#if !defined(TU_SIMD_BUFTRAITS_H)
#define TU_SIMD_BUFTRAITS_H

#include "TU/simd/allocator.h"
#include "TU/simd/load_store_iterator.h"

namespace TU
{
template <class T, class ALLOC>
class BufTraits;

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
    using element_type		= T;
    
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
    using super			= std::allocator_traits<simd::allocator<T> >;

  public:
    using iterator		= simd::iterator_wrapper<T*>;
    using const_iterator	= simd::iterator_wrapper<const T*>;
    using element_type		= typename super::value_type;
    
  protected:
    using			typename super::pointer;

    constexpr static size_t	Alignment = sizeof(simd::vec<T>);
    
    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
};

}	// namespace TU
#endif	// !TU_SIMD_BUFTRAITS_H
