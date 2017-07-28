/*!
  \file		Array++.h
  \author	Toshio UESHIBA
  \brief	SIMD命令を適用できる配列クラスの定義
*/
#if !defined(TU_SIMD_ARRAYPP_H)
#define TU_SIMD_ARRAYPP_H

#include "TU/simd/simd.h"	// import before TU/Array++.h
#if defined(SIMD)
#  include "TU/simd/algorithm.h"
#  include "TU/Array++.h"

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
    using super			= std::allocator_traits<simd::allocator<T> >;

  public:
    using iterator		= simd::iterator_wrapper<T*>;
    using const_iterator	= simd::iterator_wrapper<const T*>;
    
  protected:
    using			typename super::pointer;

    constexpr static size_t	Alignment = sizeof(simd::vec<T>);
    
    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
};

namespace simd
{
/************************************************************************
*  simd::Array<T> and simd::Array2<T> type aliases			*
************************************************************************/
template <class T, size_t N=0>
using Array  = array<T, simd::allocator<T>, N>;			//!< 1次元配列

template <class T, size_t R=0, size_t C=0>
using Array2 = array<T, simd::allocator<T>, R, C>;		//!< 2次元配列

template <class T, size_t Z=0, size_t Y=0, size_t X=0>
using Array3 = array<T, simd::allocator<T>, Z, Y, X>;		//!< 3次元配列
    
}	// namespace simd
}	// namespace TU
#else
#  include "TU/Array++.h"
#endif	// defined(SIMD)
#endif	// !TU_SIMD_ARRAYPP_H
