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
 *  $Id: iterator.h 2223 2017-01-28 02:53:29Z ueshiba $
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TU_ITERATOR_H
#define __TU_ITERATOR_H

#include <cstddef>			// for size_t
#include <type_traits>			// for std::result_of<F(ARGS...)>
#include <utility>			// for std::declval<T>
#include <iterator>
#include <boost/iterator/transform_iterator.hpp>

namespace std
{
#if __cplusplus < 201700L
/************************************************************************
*  function std::size(T)						*
************************************************************************/
template <class T> inline size_t
size(const T& x)
{
    return x.size();
}
template <class T, size_t N> inline constexpr size_t
size(const T (&array)[N]) noexcept
{
    return N;
}
#endif

#if __cplusplus < 201402L    
/************************************************************************
*  std::[rbegin|rend|cbegin|cend|crbegin|crend](T)			*
************************************************************************/
template <class T> inline auto
rbegin(const T& x) -> decltype(x.rbegin())
{
    return x.rbegin();
}
    
template <class T> inline auto
rbegin(T& x) -> decltype(x.rbegin())
{
    return x.rbegin();
}
    
template <class T> inline auto
rend(const T& x) -> decltype(x.rend())
{
    return x.rend();
}
    
template <class T> inline auto
rend(T& x) -> decltype(x.rend())
{
    return x.rend();
}
    
template <class T> inline auto
cbegin(const T& x) -> decltype(std::begin(x))
{
    return std::begin(x);
}
    
template <class T> inline auto
cend(const T& x) -> decltype(std::end(x))
{
    return std::end(x);
}

template <class T> inline auto
crbegin(const T& x) -> decltype(std::rbegin(x))
{
    return std::rbegin(x);
}
    
template <class T> inline auto
crend(const T& x) -> decltype(std::rend(x))
{
    return std::rend(x);
}
#endif
}	// namespace std

//! libTUTools++ のクラスや関数等を収める名前空間
namespace TU
{
/************************************************************************
*  type aliases								*
************************************************************************/
//! libTUTools++ のクラスや関数 の実相の詳細を収める名前空間
namespace detail
{
  template <class E>
  static auto	check_begin(const E& x) -> decltype(std::begin(x))	;
  static void	check_begin(...)					;
}	// namespace detail
    
//! 式が持つ定数反復子の型を返す
/*!
  \param E	式の型
  \return	E が定数反復子を持てばその型，持たなければ void
*/
template <class E>
using const_iterator_t = decltype(detail::check_begin(std::declval<E>()));

namespace detail
{
  template <class T>
  struct identity
  {
      using type = T;
  };

  template <class E>
  struct value_t
  {
      using type = typename std::iterator_traits<const_iterator_t<E> >
			       ::value_type;
  };
      
  template <class E, class=const_iterator_t<E> >
  struct element_t
  {
      using F	 = typename value_t<E>::type;
      using type = typename element_t<F, const_iterator_t<F> >::type;
  };
  template <class E>
  struct element_t<E, void> : identity<E>				{};
}	// namespace detail
    
//! 式が持つ定数反復子が指す型を返す
/*!
  定数反復子を持たない式を与えるとコンパイルエラーとなる.
  \param E	定数反復子を持つ式の型
  \return	E の定数反復子が指す型
*/
template <class E>
using value_t	= typename detail::value_t<E>::type;

//! 式が持つ定数反復子が指す型を再帰的に辿って到達する型を返す
/*!
  \param E	式の型
  \return	E が定数反復子を持てばそれが指す型を再帰的に辿って到達する型，
		持たなければ E 自身
*/
template <class E>
using element_t	= typename detail::element_t<E>::type;

/************************************************************************
*  transform_iterator2<FUNC, ITER0, ITER1>				*
************************************************************************/
template <class FUNC, class ITER0, class ITER1>
class transform_iterator2
    : public boost::iterator_adaptor<
		 transform_iterator2<FUNC, ITER0, ITER1>,
		 ITER0,
		 typename std::result_of<FUNC(
		     typename std::iterator_traits<ITER0>::reference,
		     typename std::iterator_traits<ITER1>::reference)>::type,
		 boost::use_default,
		 typename std::result_of<FUNC(
		     typename std::iterator_traits<ITER0>::reference,
		     typename std::iterator_traits<ITER1>::reference)>::type>
{
  private:
    using ref	= typename std::result_of<FUNC(
		      typename std::iterator_traits<ITER0>::reference,
		      typename std::iterator_traits<ITER1>::reference)>::type;
    using super	= boost::iterator_adaptor<transform_iterator2,
					  ITER0,
					  ref,
					  boost::use_default,
					  ref>;

  public:
    using	typename super::difference_type;
    using	typename super::reference;

    friend	class boost::iterator_core_access;
	
  public:
		transform_iterator2(ITER0 iter0, ITER1 iter1, const FUNC& func)
		    :super(iter0), _iter(iter1), _func(func)
		{
		}
	
  private:
    reference	dereference() const
		{
		    return _func(*super::base(), *_iter);
		}
    void	advance(difference_type n)
		{
		    super::base_reference() += n;
		    _iter += n;
		}
    void	increment()
		{
		    ++super::base_reference();
		    ++_iter;
		}
    void	decrement()
		{
		    --super::base_reference();
		    --_iter;
		}
	
  private:
    ITER1	_iter;	//!< 第2引数となる式の実体を指す反復子
    FUNC	_func;	//!< 2項演算子
};

template <class FUNC, class ITER0, class ITER1>
inline transform_iterator2<FUNC, ITER0, ITER1>
make_transform_iterator2(ITER0 iter0, ITER1 iter1, const FUNC& func)
{
    return {iter0, iter1, func};
}
    
/************************************************************************
*  alias subiterator<ITER>						*
************************************************************************/
template <class ITER>
using subiterator	= decltype(std::begin(*std::declval<ITER>()));

/************************************************************************
*  class row2col<ROW>							*
************************************************************************/
//! 行への参照を与えられると予め指定された列indexに対応する要素への参照を返す関数オブジェクト
/*!
  \param ROW	行を指す反復子
*/ 
template <class ROW>
class row2col
{
  public:
    using argument_type	= typename std::iterator_traits<ROW>::reference;
    using result_type	= typename std::iterator_traits<subiterator<ROW> >
				      ::reference;
    
  public:
		row2col(size_t col)	:_col(col)			{}
    
    result_type	operator ()(argument_type row) const
		{
		    return *(std::begin(row) + _col);
		}
    
  private:
    size_t	_col;	//!< 列を指定するindex
};

/************************************************************************
*  alias vertical_iterator<ROW>						*
************************************************************************/
template <class ROW>
using vertical_iterator = boost::transform_iterator<
			      row2col<ROW>, ROW,
			      boost::use_default,
			      typename std::iterator_traits<subiterator<ROW> >
					  ::value_type>;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(ROW row, size_t col)
{
    return {row, {col}};
}

}	// namespace TU
#endif	// !__TU_ITERATOR_H
