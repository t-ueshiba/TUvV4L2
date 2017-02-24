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
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <tuple>
#include <utility>		// std::forward(), std::index_sequence<...>
#include <iostream>
#include <boost/iterator/iterator_facade.hpp>

namespace TU
{
/************************************************************************
*  predicate is_tuple<T>, is_range_tuple<T>				*
************************************************************************/
namespace detail
{
  template <class... T>
  static std::true_type		check_tuple(std::tuple<T...>)		;
  static std::false_type	check_tuple(...)			;

  template <class T, size_t... IDX> static auto
  check_tuple(const T& x, std::index_sequence<IDX...>)
      -> decltype(std::make_tuple(std::begin(std::get<IDX>(x))...),
		  std::true_type())					;
  template <class T> static auto
  check_begin_tuple(const T& x)
      -> decltype(check_tuple(x, std::make_index_sequence<
				     std::tuple_size<T>::value>()))	;
}	// namespace detail
    
template <class T>
using is_tuple	     = decltype(detail::check_tuple(std::declval<T>()))	;
    
template <class T>
using is_range_tuple = decltype(detail::check_begin_tuple(std::declval<T>()));

/************************************************************************
*  tuple_for_each(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC> inline void
  tuple_for_each(TUPLE&, const UNARY_FUNC&, std::index_sequence<>)
  {
  }
  template <class TUPLE, class UNARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE& x, const UNARY_FUNC& f, std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x));
      tuple_for_each(x, f, std::index_sequence<IDX...>());
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC>
inline typename std::enable_if<is_tuple<TUPLE>::value>::type
tuple_for_each(TUPLE&& x, const UNARY_FUNC& f)
{
    detail::tuple_for_each(
	x, f, std::make_index_sequence<
		  std::tuple_size<typename std::decay<TUPLE>::type>::value>());
}

/************************************************************************
*  tuple_for_each(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC> inline void
  tuple_for_each(TUPLE0&, TUPLE1&, const BINARY_FUNC&, std::index_sequence<>)
  {
  }
  template <class TUPLE0, class TUPLE1,
	    class BINARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE0& x, TUPLE1& y,
		 const BINARY_FUNC& f, std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x), std::get<I>(y));
      tuple_for_each(x, y, f, std::index_sequence<IDX...>());
  }
}	// namespace detail

template <class TUPLE0, class TUPLE1, class BINARY_FUNC>
inline typename std::enable_if<TU::is_tuple<TUPLE0>::value &&
			       TU::is_tuple<TUPLE1>::value>::type
tuple_for_each(TUPLE0&& x, TUPLE1&& y, const BINARY_FUNC& f)
{
    detail::tuple_for_each(
	x, y, f, std::make_index_sequence<
	std::tuple_size<typename std::decay<TUPLE0>::type>::value>());
}

/************************************************************************
*  tuple_transform(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC, size_t... IDX>
  inline auto
  tuple_transform(TUPLE& x, const UNARY_FUNC& f, std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x))...);
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC,
	  typename std::enable_if<is_tuple<TUPLE>::value>::type* = nullptr>
inline auto
tuple_transform(TUPLE&& x, const UNARY_FUNC& f)
{
    return detail::tuple_transform(
		x, f,
		std::make_index_sequence<
		    std::tuple_size<
			typename std::decay<TUPLE>::type>::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC, size_t... IDX>
  inline auto
  tuple_transform(TUPLE0& x, TUPLE1& y,
		  const BINARY_FUNC& f, std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x), std::get<IDX>(y))...);
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class BINARY_FUNC,
	  typename std::enable_if<is_tuple<TUPLE0>::value &&
				  is_tuple<TUPLE1>::value>::type* = nullptr>
inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, const BINARY_FUNC& f)
{
    return detail::tuple_transform(
		x, y, f,
		std::make_index_sequence<
		    std::tuple_size<
			typename std::decay<TUPLE0>::type>::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, TUPLE2, TRINARY_FUNC)		*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class TUPLE2,
	    class TRINARY_FUNC, size_t... IDX> inline auto
  tuple_transform(TUPLE0& x, TUPLE1& y, TUPLE2& z,
		  const TRINARY_FUNC& f, std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x), std::get<IDX>(y),
			       std::get<IDX>(z))...);
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class TUPLE2, class TRINARY_FUNC,
	  typename std::enable_if<is_tuple<TUPLE0>::value &&
				  is_tuple<TUPLE1>::value &&
				  is_tuple<TUPLE2>::value>::type* = nullptr>
inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, TUPLE2&& z, const TRINARY_FUNC& f)
{
    return detail::tuple_transform(
		x, y, z, f,
		std::make_index_sequence<
		    std::tuple_size<
			typename std::decay<TUPLE0>::type>::value>());
}

}	// namespace TU

/*
 *  argument dependent lookup が働くために，std::tuple<...>を引数とする
 *  operator overloadされた関数は namespace std 中に定義しなければならない．
 */
namespace std
{
/************************************************************************
*  I/O functions							*
************************************************************************/
template <class TUPLE>
inline typename enable_if<TU::is_tuple<TUPLE>::value, ostream&>::type
operator <<(ostream& out, const TUPLE& t)
{
    out << '(';
    TU::tuple_for_each(t, [&](const auto& x){ out << ' ' << x; });
    out << ')';

    return out;
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class TUPLE,
	  typename enable_if<TU::is_tuple<TUPLE>::value>::type* = nullptr>
inline auto
operator -(const TUPLE& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return -x; });
}

template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator +(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x + y; });
}

template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator -(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x - y; });
}

template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x * y; });
}

template <class TUPLE, class T,
	  typename enable_if<TU::is_tuple<TUPLE>::value &&
			     !TU::is_tuple<T>::value>::type* = nullptr>
inline auto
operator *(const TUPLE& t, const T& c)
{
    return TU::tuple_transform(t, [&](const auto& x){ return x * c; });
}

template <class T, class TUPLE,
	  typename enable_if<!TU::is_tuple<T>::value &&
			     TU::is_tuple<TUPLE>::value>::type* = nullptr>
inline auto
operator *(const T& c, const TUPLE& t)
{
    return TU::tuple_transform(t, [&](const auto& x){ return c * x; });
}

template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator /(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x / y; });
}

template <class TUPLE, class T,
	  typename enable_if<TU::is_tuple<TUPLE>::value &&
			     !TU::is_tuple<T>::value>::type* = nullptr>
inline auto
operator /(const TUPLE& t, const T& c)
{
    return TU::tuple_transform(t, [&](const auto& x){ return x / c; });
}

template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator %(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x % y; });
}

template <class TUPLE, class T,
	  typename enable_if<TU::is_tuple<TUPLE>::value &&
			     !TU::is_tuple<T>::value>::type* = nullptr>
inline auto
operator %(const TUPLE& t, const T& c)
{
    return TU::tuple_transform(t, [&](const auto& x){ return x % c; });
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator +=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x += y; });
    return l;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator -=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x -= y; });
    return l;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator *=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x *= y; });
    return l;
}

template <class TUPLE, class T>
inline typename enable_if<TU::is_tuple<TUPLE>::value &&
			  !TU::is_tuple<T>::value, TUPLE&>::type
operator *=(TUPLE&& t, const T& c)
{
    TU::tuple_for_each(forward<TUPLE>(t), [&](auto& x){ x *= c; });
    return t;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator /=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x /= y; });
    return l;
}

template <class TUPLE, class T>
inline typename enable_if<TU::is_tuple<TUPLE>::value &&
			  !TU::is_tuple<T>::value, TUPLE&>::type
operator /=(TUPLE&& t, const T& c)
{
    TU::tuple_for_each(forward<TUPLE>(t), [&](auto& x){ x /= c; });
    return t;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator %=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x %= y; });
    return l;
}

template <class TUPLE, class T>
inline typename enable_if<TU::is_tuple<TUPLE>::value &&
			  !TU::is_tuple<T>::value, TUPLE&>::type
operator %=(TUPLE&& t, const T& c)
{
    TU::tuple_for_each(forward<TUPLE>(t), [&](auto& x){ x %= c; });
    return t;
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator &(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x & y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator |(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x | y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator ^(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x ^ y; });
}
    
template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator &=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x &= y; });
    return l;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator |=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x |= y; });
    return l;
}

template <class L, class R>
inline typename enable_if<TU::is_tuple<L>::value &&
			  TU::is_tuple<R>::value, L&>::type
operator ^=(L&& l, const R& r)
{
    TU::tuple_for_each(forward<L>(l), r, [](auto& x, const auto& y){ x ^= y; });
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class TUPLE,
	  typename enable_if<TU::is_tuple<TUPLE>::value>::type* = nullptr>
inline auto
operator !(const TUPLE& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return !x; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator &&(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x && y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator ||(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x || y; });
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator ==(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x == y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator !=(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x != y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator <(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x < y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator >(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x > y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator <=(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x <= y; });
}
    
template <class L, class R,
	  typename enable_if<TU::is_tuple<L>::value &&
			     TU::is_tuple<R>::value>::type* = nullptr>
inline auto
operator >=(const L& l, const R& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x >= y; });
}

}	// namespace std

namespace TU
{
/************************************************************************
*  Selection								*
************************************************************************/
template <class S, class X, class Y,
	  typename std::enable_if<is_tuple<S>::value &&
				  is_tuple<X>::value &&
				  is_tuple<Y>::value>::type* = nullptr>
inline auto
select(const S& s, const X& x, const Y& y)
{
    return tuple_transform(s, x, y,
			   [](const auto& t, const auto& u, const auto& v)
			   { return select(t, u, v); });
}

template <class S, class X, class Y,
	  typename std::enable_if<is_tuple<S>::value &&
				  !is_tuple<X>::value &&
				  is_tuple<Y>::value>::type* = nullptr>
inline auto
select(const S& s, const X& x, const Y& y)
{
    return tuple_transform(s, y, [&](const auto& t, const auto& v)
				 { return select(t, x, v); });
}

template <class S, class X, class Y,
	  typename std::enable_if<is_tuple<S>::value &&
				  is_tuple<X>::value &&
				  !is_tuple<Y>::value>::type* = nullptr>
inline auto
select(const S& s, const X& x, const Y& y)
{
    return tuple_transform(s, x, [&](const auto& t, const auto& u)
				 { return select(t, u, y); });
}

/************************************************************************
*  class unarizer<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって多変数関数を1変数関数に変換
template <class FUNC>
class unarizer
{
  public:
    using functor_type = FUNC;

  public:
    unarizer(const FUNC& func=FUNC())	:_func(func)	{}

    template <class TUPLE_,
	      typename std::enable_if<is_tuple<TUPLE_>::value>::type* = nullptr>
    auto	operator ()(const TUPLE_& arg) const
		{
		    return exec(arg, std::make_index_sequence<
					 std::tuple_size<TUPLE_>::value>());
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    template <class TUPLE_, size_t... IDX_>
    auto	exec(const TUPLE_& arg, std::index_sequence<IDX_...>) const
		{
		    return _func(std::get<IDX_>(arg)...);
		}

  private:
    FUNC	_func;
};

template <class FUNC> inline unarizer<FUNC>
make_unarizer(const FUNC& func)
{
    return {func};
}
    
/************************************************************************
*  class zip_iterator<ITER_TUPLE>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
      template <class ITER> 
      typename std::enable_if<
	  !std::is_reference<
	      typename std::iterator_traits<ITER>::reference>::value,
	  typename std::iterator_traits<ITER>::reference>::type
      operator ()(const ITER& iter) const
      {
	  return *iter;
      }
      template <class ITER> auto
      operator ()(const ITER& iter) const
	  -> typename std::enable_if<
	      std::is_reference<
		  typename std::iterator_traits<ITER>::reference>::value,
	      decltype(std::ref(*iter))>::type
      {
	  return std::ref(*iter);
      }
  };
}	// namespace detail
    
template <class ITER_TUPLE>
class zip_iterator
    : public boost::iterator_facade<
	  zip_iterator<ITER_TUPLE>,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference())),
	  typename std::iterator_traits<
	      typename std::tuple_element<0, ITER_TUPLE>::type>
	      ::iterator_category,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference()))>
{
  private:
    using super = boost::iterator_facade<
		      zip_iterator,
		      decltype(
			  tuple_transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference())),
		      typename std::iterator_traits<
			  typename std::tuple_element<0, ITER_TUPLE>::type>
			  ::iterator_category,
		      decltype(
			  tuple_transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference()))>;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;
    
    friend class	boost::iterator_core_access;

  private:
    struct Increment
    {
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ ++iter; }
    };

    struct Decrement
    {
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ --iter; }
    };

    struct Advance
    {
	Advance(difference_type n)	:_n(n)	{}
	
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ std::advance(iter, _n); }

      private:
	const difference_type	_n;
    };

  public:
    zip_iterator(const ITER_TUPLE& iter_tuple)
	:_iter_tuple(iter_tuple)		{}

    const ITER_TUPLE&
		get_iterator_tuple()	const	{ return _iter_tuple; }
    
  private:
    reference	dereference() const
		{
		    return tuple_transform(_iter_tuple,
					   detail::generic_dereference());
		}
    bool	equal(const zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			== std::get<0>(_iter_tuple);
		}
    void	increment()
		{
		    tuple_for_each(_iter_tuple, Increment());
		}
    void	decrement()
		{
		    tuple_for_each(_iter_tuple, Decrement());
		}
    void	advance(difference_type n)
		{
		    tuple_for_each(_iter_tuple, Advance(n));
		}
    difference_type
		distance_to(const zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			 - std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class ITER_TUPLE> inline zip_iterator<ITER_TUPLE>
make_zip_iterator(const ITER_TUPLE& iter_tuple)
{
    return {iter_tuple};
}

}	// namespace TU

namespace std
{
/************************************************************************
*  std::[begin|end|rbegin|rend](std::tuple<T...>)			*
************************************************************************/
namespace detail
{
  struct generic_begin
  {
      template <class T>
      auto	operator ()(T&& x)	const	{ return std::begin(x); }
  };
    
  struct generic_end
  {
      template <class T>
      auto	operator ()(T&& x)	const	{ return std::end(x); }
  };

  struct generic_rbegin
  {
      template <class T>
      auto	operator ()(T&& x)	const	{ return std::rbegin(x); }
  };
    
  struct generic_rend
  {
      template <class T>
      auto	operator ()(T&& x)	const	{ return std::rend(x); }
  };
}	// namespace detail

template <class TUPLE,
	  typename enable_if<TU::is_range_tuple<TUPLE>::value>::type* = nullptr>
inline auto
begin(TUPLE&& x)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     x, detail::generic_begin()));
}

template <class TUPLE,
	  typename enable_if<TU::is_range_tuple<TUPLE>::value>::type* = nullptr>
inline auto
end(TUPLE&& x)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     x, detail::generic_end()));
}
    
template <class TUPLE,
	  typename enable_if<TU::is_range_tuple<TUPLE>::value>::type* = nullptr>
inline auto
rbegin(TUPLE&& x)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     x, detail::generic_rbegin()));
}
    
template <class TUPLE,
	  typename enable_if<TU::is_range_tuple<TUPLE>::value>::type* = nullptr>
inline auto
rend(TUPLE&& x)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     x, detail::generic_rend()));
}

}	// namespace std
#endif	// !__TU_TUPLE_H
