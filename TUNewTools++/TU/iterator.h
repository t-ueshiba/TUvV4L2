/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TU_ITERATOR_H
#define __TU_ITERATOR_H

#include <cstddef>			// for size_t
#include <type_traits>			// for std::result_of<F(ARGS...)>
#include <utility>			// for std::declval<T>
#include <functional>			// for std::function
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
template <class ITER>
using iterator_value	  = typename std::iterator_traits<ITER>::value_type;
template <class ITER>
using iterator_reference  = typename std::iterator_traits<ITER>::reference;
template <class ITER>
using iterator_pointer	  = typename std::iterator_traits<ITER>::pointer;
template <class ITER>
using iterator_difference = typename std::iterator_traits<ITER>
					::difference_type;
template <class ITER>
using iterator_category	  = typename std::iterator_traits<ITER>
					::iterator_category;

//! libTUTools++ のクラスや関数の実装の詳細を収める名前空間
namespace detail
{
  template <class T>
  auto	check_begin(const T& x) -> decltype(std::begin(x))		;
  void	check_begin(...)						;
}

//! 式が持つ定数反復子の型を返す
/*!
  \param E	式の型
  \return	E が定数反復子を持てばその型，持たなければ void
*/

template <class E>
using const_iterator_t	= decltype(detail::check_begin(std::declval<E>()));

//! 式が定数反復子を持つか判定する
template <class T>
using has_begin		= std::integral_constant<
			      bool, !std::is_void<const_iterator_t<T> >::value>;
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
      using type = iterator_value<const_iterator_t<E> >;
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
*  make_mbr_iterator<ITER, T>						*
************************************************************************/
//! T型のメンバ変数を持つオブジェクトへの反復子からそのメンバに直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
  \param mbr	iterが指すオブジェクトのメンバへのポインタ
*/
template <class ITER, class T> inline auto
make_mbr_iterator(const ITER& iter, T iterator_value<ITER>::* mbr)
{
    return boost::make_transform_iterator(
	       iter,
	       std::function<std::conditional_t<
				 std::is_same<iterator_pointer<ITER>,
					      iterator_value<ITER>*>::value,
				 T&, const T&>(iterator_reference<ITER>)>(
				     std::mem_fn(mbr)));
}

//! std::pairへの反復子からその第1要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_first_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter,
			     &std::iterator_traits<ITER>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_second_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter,
			     &std::iterator_traits<ITER>::value_type::second);
}
    
/************************************************************************
*  transform_iterator2<FUNC, ITER0, ITER1>				*
************************************************************************/
template <class FUNC, class ITER0, class ITER1>
class transform_iterator2
    : public boost::iterator_adaptor<
		 transform_iterator2<FUNC, ITER0, ITER1>,
		 ITER0,
		 typename std::result_of<FUNC(iterator_reference<ITER0>,
					      iterator_reference<ITER1>)>::type,
		 boost::use_default,
		 typename std::result_of<FUNC(iterator_reference<ITER0>,
					      iterator_reference<ITER1>)>::type>
{
  private:
    using ref	= typename std::result_of<FUNC(iterator_reference<ITER0>,
					       iterator_reference<ITER1>)>::type;
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
		transform_iterator2(ITER0 iter0, ITER1 iter1, FUNC func)
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
make_transform_iterator2(ITER0 iter0, ITER1 iter1, FUNC func)
{
    return {iter0, iter1, func};
}
    
/************************************************************************
*  class assignment_iterator<FUNC, ITER>				*
************************************************************************/
namespace detail
{
  template <class FUNC, class ITER>
  class assignment_proxy
  {
    public:
      using	self = assignment_proxy;
	
    public:
      assignment_proxy(const ITER& iter, const FUNC& func)
	  :_iter(iter), _func(func)					{}

      template <class T_>
      self&	operator =(const T_& val)
		{
		    _func(*_iter, val);
		    return *this;
		}

    private:
      const ITER&	_iter;
      const FUNC&	_func;
  };
}

//! operator *()を左辺値として使うときに，この左辺値と右辺値に指定された関数を適用するための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class ITER>
class assignment_iterator
    : public boost::iterator_adaptor<assignment_iterator<FUNC, ITER>,
				     ITER,
				     detail::assignment_proxy<FUNC, ITER>,
				     boost::use_default,
				     detail::assignment_proxy<FUNC, ITER> >
{
  private:
    using super	= boost::iterator_adaptor<
		      assignment_iterator,
		      ITER,
		      detail::assignment_proxy<FUNC, ITER>,
		      boost::use_default,
		      detail::assignment_proxy<FUNC, ITER> >;

  public:
    using		typename super::reference;
    
    friend class	boost::iterator_core_access;

  public:
    assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)			{}

    const FUNC&	functor()			const	{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return {super::base(), _func};
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
{
    return {iter, func};
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
    using argument_type	= iterator_reference<ROW>;
    using result_type	= iterator_reference<subiterator<ROW> >;
    
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
			      iterator_value<subiterator<ROW> > >;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(ROW row, size_t col)
{
    return {row, {col}};
}

}	// namespace TU
#endif	// !__TU_ITERATOR_H
