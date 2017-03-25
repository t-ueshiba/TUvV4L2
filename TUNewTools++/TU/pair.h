/*
 *  $Id: functional.h 1775 2014-12-24 06:08:59Z ueshiba $
 */
/*!
  \file		pair.h
  \brief	std::pairの用途拡張のためのユティリティ
*/
#ifndef __TU_PAIR_H
#define __TU_PAIR_H

#include <utility>
#include <iostream>

namespace TU
{
/************************************************************************
*  struct is_pair<T>							*
************************************************************************/
namespace detail
{
  template <class S, class T>
  std::true_type	check_pair(std::pair<S, T>)			;
  std::false_type	check_pair(...)					;
}	// namespace detail

template <class T>
using is_pair = decltype(detail::check_pair(std::declval<T>()));

/************************************************************************
*  struct pair_traits<PAIR>						*
************************************************************************/
template <class T>
struct pair_traits
{
    static constexpr size_t	size = 1;
    using leftmost_type		= T;
    using rightmost_type	= T;
};
template <class S, class T>
struct pair_traits<std::pair<S, T> >
{
    constexpr static size_t	size = pair_traits<S>::size
				     + pair_traits<T>::size;
    using leftmost_type		= typename pair_traits<S>::leftmost_type;
    using rightmost_type	= typename pair_traits<S>::rightmost_type;
};

/************************************************************************
*  struct pair_tree<T, N>						*
************************************************************************/
namespace detail
{
  template <class T, size_t N>
  struct pair_tree
  {
      using type	= std::pair<typename pair_tree<T, (N>>1)>::type,
				    typename pair_tree<T, (N>>1)>::type>;
  };
  template <class T>
  struct pair_tree<T, 1>
  {
      using type	= T;
  };
  template <class T>
  struct pair_tree<T, 0>
  {
      using type	= T;
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
