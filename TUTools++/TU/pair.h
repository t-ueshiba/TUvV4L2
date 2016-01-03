/*
 *  平成14-24年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2012.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: functional.h 1775 2014-12-24 06:08:59Z ueshiba $
 */
/*!
  \file		tuple.h
  \brief	boost::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_PAIR_H
#define __TU_PAIR_H

#include <utility>
#include <iostream>
#include "TU/functional.h"	// is_convertible<T, C<ARGS...> >

namespace TU
{
/************************************************************************
*  struct is_pair<T>							*
************************************************************************/
template <class T>
using is_pair = is_convertible<T, std::pair>;

/************************************************************************
*  struct pair_traits<PAIR>						*
************************************************************************/
template <class T>
struct pair_traits
{
    static constexpr size_t	size = 1;
    typedef T						leftmost_type;
    typedef T						rightmost_type;
};
template <class S, class T>
struct pair_traits<std::pair<S, T> >
{
    static constexpr size_t	size = pair_traits<S>::size
				     + pair_traits<T>::size;
    typedef typename pair_traits<S>::leftmost_type	leftmost_type;
    typedef typename pair_traits<T>::rightmost_type	rightmost_type;
};

/************************************************************************
*  struct pair_tree<T, N>						*
************************************************************************/
namespace detail
{
  template <class T, size_t N>
  struct pair_tree
  {
      typedef std::pair<typename pair_tree<T, (N>>1)>::type,
			typename pair_tree<T, (N>>1)>::type>	type;
  };
  template <class T>
  struct pair_tree<T, 1>
  {
      typedef T							type;
  };
  template <class T>
  struct pair_tree<T, 0>
  {
      typedef T							type;
  };
}	// namespace detail

//! 2のべき乗個の同一要素から成る多層pairを表すクラス
/*!
  \param T	要素の型
  \param N	要素の個数(2のべき乗)
*/
template <class T, size_t N=1>
using pair_tree = typename detail::pair_tree<T, N>::type;
    
}	// namespace TU

namespace std
{
template <class S, class T> inline ostream&
operator <<(ostream& out, const std::pair<S, T>& x)
{
    return out << '[' << x.first << ' ' << x.second << ']';
}
    
}	// namespace std

#endif	// !__TU_PAIR_H
