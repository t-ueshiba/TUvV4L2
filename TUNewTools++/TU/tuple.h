/*!
  \file		tuple.h
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <tuple>
#include <iostream>
#include "TU/iterator.h"

namespace TU
{
/************************************************************************
*  predicates all<PRED, T>, is_tuple<T>, all_has_begin<T>		*
************************************************************************/
template <template <class> class PRED, class T>
struct all;
template <template <class> class PRED>
struct all<PRED, std::tuple<> >
{
    constexpr static bool	value = true;
};
template <template <class> class PRED, class HEAD, class... TAIL>
struct all<PRED, std::tuple<HEAD, TAIL...> >
{
    constexpr static bool	value = PRED<HEAD>::value &&
					all<PRED, std::tuple<TAIL...> >::value;
};

namespace detail
{
  template <class... T>
  std::true_type	check_tuple(std::tuple<T...>)			;
  std::false_type	check_tuple(...)				;
}	// namespace detail
    
template <class T>
using is_tuple		= decltype(detail::check_tuple(std::declval<T>()));

template <class T>
using all_has_begin	= all<has_begin, std::decay_t<T> >;

/************************************************************************
*  tuple_for_each(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC> inline void
  tuple_for_each(TUPLE&&, UNARY_FUNC, std::index_sequence<>)
  {
  }
  template <class TUPLE, class UNARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE&& x, UNARY_FUNC f, std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x));
      tuple_for_each(std::forward<TUPLE>(x), f, std::index_sequence<IDX...>());
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC>
inline std::enable_if_t<is_tuple<TUPLE>::value>
tuple_for_each(TUPLE&& x, UNARY_FUNC f)
{
    detail::tuple_for_each(
	std::forward<TUPLE>(x), f,
	std::make_index_sequence<std::tuple_size<std::decay_t<TUPLE> >
	   ::value>());
}

/************************************************************************
*  make_reference_wrapper(T&&)						*
************************************************************************/
//! 与えられた値の型に応じて実引数を生成する
/*!
  \param x	関数に渡す引数
  \return	xの型Tが参照でなければx自身，定数参照ならばstd::cref(x),
		非定数参照ならばstd::ref(x)
*/
template <class T>
inline std::conditional_t<std::is_reference<T>::value,
			  std::reference_wrapper<std::remove_reference_t<T> >,
			  T>
make_reference_wrapper(T&& x)
{
    return x;
}
    
/************************************************************************
*  tuple_for_each(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC> inline void
  tuple_for_each(TUPLE0&&, TUPLE1&&, BINARY_FUNC, std::index_sequence<>)
  {
  }
  template <class TUPLE0, class TUPLE1,
	    class BINARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f,
		 std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x), std::get<I>(y));
      tuple_for_each(std::forward<TUPLE0>(x), std::forward<TUPLE1>(y), f,
		     std::index_sequence<IDX...>());
  }
}	// namespace detail

template <class TUPLE0, class TUPLE1, class BINARY_FUNC>
inline std::enable_if_t<is_tuple<TUPLE0>::value && is_tuple<TUPLE1>::value>
tuple_for_each(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f)
{
    detail::tuple_for_each(
	std::forward<TUPLE0>(x), std::forward<TUPLE1>(y), f,
	std::make_index_sequence<std::tuple_size<std::decay_t<TUPLE0> >
	   ::value>());
}

/************************************************************************
*  tuple_transform(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC, size_t... IDX> inline auto
  tuple_transform(TUPLE&& x, UNARY_FUNC f, std::index_sequence<IDX...>)
  {
      return std::make_tuple(make_reference_wrapper(f(std::get<IDX>(x)))...);
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
tuple_transform(TUPLE&& x, UNARY_FUNC f)
{
    return detail::tuple_transform(
		std::forward<TUPLE>(x), f,
		std::make_index_sequence<std::tuple_size<std::decay_t<TUPLE> >
		   ::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC, size_t... IDX>
  inline auto
  tuple_transform(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f,
		  std::index_sequence<IDX...>)
  {
      return std::make_tuple(make_reference_wrapper(
				 f(std::get<IDX>(x), std::get<IDX>(y)))...);
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class BINARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE0>::value &&
			   is_tuple<TUPLE1>::value>* = nullptr> inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f)
{
    return detail::tuple_transform(
		std::forward<TUPLE0>(x), std::forward<TUPLE1>(y), f,
		std::make_index_sequence<std::tuple_size<std::decay_t<TUPLE0> >
		   ::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, TUPLE2, TRINARY_FUNC)		*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class TUPLE2, class TRINARY_FUNC,
	    size_t... IDX> inline auto
  tuple_transform(TUPLE0&& x, TUPLE1&& y, TUPLE2&& z, TRINARY_FUNC f,
		  std::index_sequence<IDX...>)
  {
      return std::make_tuple(make_reference_wrapper(
				 f(std::get<IDX>(x), std::get<IDX>(y),
				   std::get<IDX>(z))...));
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class TUPLE2, class TRINARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE0>::value &&
			   is_tuple<TUPLE1>::value &&
			   is_tuple<TUPLE2>::value>* = nullptr> inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, TUPLE2&& z, TRINARY_FUNC f)
{
    return detail::tuple_transform(
		std::forward<TUPLE0>(x), std::forward<TUPLE1>(y),
		std::forward<TUPLE2>(z), f,
		std::make_index_sequence<std::tuple_size<std::decay_t<TUPLE0> >
		   ::value>());
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> inline auto
select(bool s, const X& x, const Y& y)
{
    return (s ? x : y);
}
    
template <class... S, class... X, class... Y> inline auto
select(const std::tuple<S...>& s,
       const std::tuple<X...>& x, const std::tuple<Y...>& y)
{
    return tuple_transform(s, x, y,
			   [](const auto& t, const auto& u, const auto& v)
			   { return select(t, u, v); });
}

template <class... S, class X, class... Y> inline auto
select(const std::tuple<S...>& s, const X& x, const std::tuple<Y...>& y)
{
    return tuple_transform(s, y, [&x](const auto& t, const auto& v)
				 { return select(t, x, v); });
}

template <class... S, class... X, class Y> inline auto
select(const std::tuple<S...>& s, const std::tuple<X...>& x, const Y& y)
{
    return tuple_transform(s, x, [&y](const auto& t, const auto& u)
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
    unarizer(FUNC func=FUNC())	:_func(func)		{}

    template <class TUPLE_,
	      std::enable_if_t<is_tuple<TUPLE_>::value>* = nullptr>
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
    const FUNC	_func;
};

template <class FUNC> inline unarizer<FUNC>
make_unarizer(FUNC func)
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
      template <class ITER_>
      decltype(auto)	operator ()(ITER_ iter)	const	{ return *iter; }
  };
}	// namespace detail
    
template <class ITER_TUPLE>
class zip_iterator
    : public boost::iterator_facade<
	  zip_iterator<ITER_TUPLE>,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference())),
	  iterator_category<typename std::tuple_element<0, ITER_TUPLE>::type>,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference()))>
{
  private:
    using super = boost::iterator_facade<
		      zip_iterator,
		      decltype(tuple_transform(std::declval<ITER_TUPLE>(),
					       detail::generic_dereference())),
		      iterator_category<
			  typename std::tuple_element<0, ITER_TUPLE>::type>,
		      decltype(tuple_transform(std::declval<ITER_TUPLE>(),
					       detail::generic_dereference()))>;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;
    
    friend class	boost::iterator_core_access;

  public:
    zip_iterator(ITER_TUPLE iter_tuple)
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
		    tuple_for_each(_iter_tuple, [](auto& x){ ++x; });
		}
    void	decrement()
		{
		    tuple_for_each(_iter_tuple, [](auto& x){ --x; });
		}
    void	advance(difference_type n)
		{
		    tuple_for_each(_iter_tuple, [n](auto& x){ x += n; });
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
make_zip_iterator(ITER_TUPLE iter_tuple)
{
    return {iter_tuple};
}

/************************************************************************
*  struct tuple_head<T>, tuple_leftmost<T>, tuple_nelms<T>		*
************************************************************************/
namespace detail
{
  template <class T>
  struct tuple_traits
  {
      static constexpr size_t	nelms = 1;
      using head_type		= T;
      using leftmost_type	= T;
  };
  template <>
  struct tuple_traits<std::tuple<> >
  {
      static constexpr size_t	nelms = 0;
      using head_type		= void;
      using leftmost_type	= void;
  };
  template <class HEAD, class... TAIL>
  struct tuple_traits<std::tuple<HEAD, TAIL...> >
  {
      static constexpr size_t	nelms = tuple_traits<HEAD>::nelms
				      + tuple_traits<
					    std::tuple<TAIL...> >::nelms;
      using head_type		= HEAD;
      using leftmost_type	= typename tuple_traits<HEAD>::leftmost_type;
  };
}
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = typename detail::tuple_traits<T>::head_type;

//! 与えられた型がtupleならばその最左要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その最左要素の型を調べるべき型
*/
template <class T>
using tuple_leftmost = typename detail::tuple_traits<T>::leftmost_type;

//! 与えられた型がtupleまたはnull_typeならばその要素数を，そうでなければ1を返す．
/*!
  \param T	要素数を調べるべき型
*/
template <class T>
struct tuple_nelms
{
    static constexpr size_t	value = detail::tuple_traits<T>::nelms;
};
    
/************************************************************************
*  struct tuple_for_all<T, COND, ARGS...>				*
************************************************************************/
template <class T, template <class...> class COND, class... ARGS>
struct tuple_for_all : std::integral_constant<bool, COND<T, ARGS...>::value>
{
};
template <template <class...> class COND, class... ARGS>
struct tuple_for_all<std::tuple<>, COND, ARGS...> : std::true_type
{
};
template <class HEAD, class... TAIL,
	  template <class...> class COND, class... ARGS>
struct tuple_for_all<std::tuple<HEAD, TAIL...>, COND,  ARGS...>
    : std::integral_constant<
	  bool, (COND<HEAD, ARGS...>::value &&
		 tuple_for_all<std::tuple<TAIL...>, COND, ARGS...>::value)>
{
};

/************************************************************************
*  struct tuple_is_uniform<T>						*
************************************************************************/
template <class T>
struct tuple_is_uniform : std::true_type
{
};
template <class T>
struct tuple_is_uniform<std::tuple<T> > : std::true_type
{
};
template <class HEAD, class... TAIL>
struct tuple_is_uniform<std::tuple<HEAD, TAIL...> >
    : std::integral_constant<
	  bool,
	  (std::is_same<HEAD, tuple_head<std::tuple<TAIL...> > >::value &&
	   tuple_is_uniform<std::tuple<TAIL...> >::value)>
{
};
    
/************************************************************************
*  struct tuple_replace<S, T>						*
************************************************************************/
namespace detail
{
  template <class T, class S>
  struct tuple_replace : std::conditional<std::is_void<T>::value, S, T>
  {
  };
  template <class T, class... S>
  struct tuple_replace<T, std::tuple<S...> >
  {
      using type = std::tuple<typename tuple_replace<T, S>::type...>;
  };
}
    
//! 与えられた型がtupleならばその全要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T=void>
using tuple_replace = typename detail::tuple_replace<T, S>::type;

}	// namespace TU

/*
 *  argument dependent lookup が働くために，std::tuple<...>を引数とする
 *  operator overloadされた関数は namespace std 中に定義しなければならない．
 */
namespace std
{
/************************************************************************
*  std::[begin|end|rbegin|rend](std::tuple<T...>)			*
************************************************************************/
/*
 *  icpc-17.0.2 のバグ回避のため，lambda関数ではなくgenericな関数オブジェクトを
 *  用いて実装
 */ 
namespace detail
{
  struct generic_begin
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::begin(x); }
  };
  struct generic_end
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::end(x); }
  };
  struct generic_rbegin
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::rbegin(x); }
  };
  struct generic_rend
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::rend(x); }
  };
}	// namespace detail
    
template <class TUPLE, enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr>
inline auto
begin(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_begin()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return begin(x); }));
}

template <class TUPLE, enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr>
inline auto
end(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_end()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return end(x); }));
}
    
template <class TUPLE, enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr>
inline auto
rbegin(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_rbegin()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return rbegin(x); }));
}

template <class TUPLE, enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr>
inline auto
rend(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_rend()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return rend(x); }));
}
    
template <class... T> inline size_t
size(const tuple<T...>& t)
{
    return size(get<0>(t));
}
    
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class... T> inline auto
operator -(const tuple<T...>& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return -x; });
}

template <class... L, class... R> inline auto
operator +(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x + y; });
}

template <class... L, class... R> inline auto
operator -(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x - y; });
}

template <class... L, class... R> inline auto
operator *(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x * y; });
}

template <class... L, class T> inline auto
operator *(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x * c; });
}

template <class T, class... R> inline auto
operator *(const T& c, const tuple<R...>& r)
{
    return TU::tuple_transform(r, [&c](const auto& x){ return c * x; });
}

template <class... L, class... R> inline auto
operator /(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x / y; });
}

template <class... L, class T> inline auto
operator /(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x / c; });
}

template <class... L, class... R> inline auto
operator %(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x % y; });
}

template <class... L, class T> inline auto
operator %(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x % c; });
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator +=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x += y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator -=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x -= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator *=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x *= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator *=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x *= c; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator /=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x /= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator /=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x /= c; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator %=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x %= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator %=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x %= c; });
    return l;
}

template <class... L, class... C, class... R> inline auto
fma(const tuple<L...>& l, const tuple<C...>& c, const tuple<R...>& r)
{
    return TU::tuple_transform(l, c, r,
			       [](const auto& x, const auto& y, const auto& z)
			       { return fma(x, y, z); });
}

template <class T, class... L, class... R> inline auto
fma(const T& c, const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [&c](const auto& x, const auto& y)
				     { return fma(c, x, y); });
}

template <class... L, class T, class... R> inline auto
fma(const tuple<L...>& l, const T& c, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [&c](const auto& x, const auto& y)
				     { return fma(x, c, y); });
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class... L, class... R> inline auto
operator &(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x & y; });
}
    
template <class... L, class... R> inline auto
operator |(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x | y; });
}
    
template <class... L, class... R> inline auto
operator ^(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x ^ y; });
}
    
template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator &=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x &= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator |=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x |= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator ^=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x ^= y; });
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> inline auto
operator !(const tuple<T...>& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return !x; });
}
    
template <class... L, class... R> inline auto
operator &&(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x && y; });
}
    
template <class... L, class... R> inline auto
operator ||(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x || y; });
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class... L, class... R> inline auto
operator ==(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x == y; });
}
    
template <class... L, class... R> inline auto
operator !=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x != y; });
}
    
template <class... L, class... R> inline auto
operator <(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x < y; });
}
    
template <class... L, class... R> inline auto
operator >(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x > y; });
}
    
template <class... L, class... R> inline auto
operator <=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x <= y; });
}
    
template <class... L, class... R> inline auto
operator >=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x >= y; });
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class... T> inline ostream&
operator <<(ostream& out, const tuple<T...>& t)
{
    out << '(';
    TU::tuple_for_each(t, [&out](const auto& x){ out << ' ' << x; });
    out << ')';

    return out;
}


}	// namespace std
#endif	// !__TU_TUPLE_H
