/*!
  \file		tuple.h
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <tuple>
#include <iostream>
#include "TU/iterator.h"	// for std::rbegin(), std::rend() before C++14

namespace TU
{
/************************************************************************
*  predicates: all<PRED, TUPLE>						*
************************************************************************/
//! std::tuple の全要素型が指定された条件を満たすか判定する
/*!
  \param PRED	適用する述語
  \param TUPLE	適用対象となる std::tuple
*/
template <template <class> class PRED, class TUPLE>
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

/************************************************************************
*  predicates: has_begin<T>, all_has_begin<T>				*
************************************************************************/
namespace detail
{
  template <class T>
  auto	has_begin(T&& x) -> decltype(std::begin(x), std::true_type())	;
  auto	has_begin(...)	 -> std::false_type				;
}	// namespace detail

//! 与えられた型に std::begin() を適用できるか判定する
/*!
  \param T	判定対象となる型
*/ 
template <class T>
using has_begin		= decltype(detail::has_begin(std::declval<T>()));

//! 与えられた std::tuple の全要素の型に std::begin() を適用できるか判定する
/*!
  \param TUPLE	判定対象となる std::tuple 型
*/ 
template <class TUPLE>
using all_has_begin	= all<has_begin, std::decay_t<TUPLE> >;

/************************************************************************
*  predicate: is_tuple<T>						*
************************************************************************/
namespace detail
{
  template <class... T>
  std::true_type	check_tuple(std::tuple<T...>)			;
  std::false_type	check_tuple(...)				;
}	// namespace detail

//! 与えられた型が std::tuple であるか判定する
/*!
  \param T	判定対象となる型
*/ 
template <class T>
using is_tuple		= decltype(detail::check_tuple(std::declval<T>()));

/************************************************************************
*  predicate: any_tuple<ARGS...>					*
************************************************************************/
//! 少なくとも1つのテンプレート引数が std::tuple であるか判定する
/*!
  \param ARGS...	判定対象となる型の並び
*/
template <class... ARGS>
struct any_tuple : std::false_type					{};
template <class ARG, class... ARGS>
struct any_tuple<ARG, ARGS...>
    : std::integral_constant<bool, (is_tuple<ARG>::value ||
				    any_tuple<ARGS...>::value)>		{};
    
/************************************************************************
*  type alias: tuple_head<T>						*
************************************************************************/
namespace detail
{
  template <class T>
  struct tuple_head
  {
      using type = T;
  };
  template <class... T>
  struct tuple_head<std::tuple<T...> >
  {
      using type = std::tuple_element_t<0, std::tuple<T...> >;
  };
}	// namespace detail
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = typename detail::tuple_head<T>::type;

/************************************************************************
*  type alias: tuple_replace<S, T>					*
************************************************************************/
namespace detail
{
  template <class S, class T>
  struct tuple_replace : std::conditional<std::is_void<T>::value, S, T>
  {
  };
  template <class... S, class T>
  struct tuple_replace<std::tuple<S...>, T>
  {
      using type = std::tuple<typename tuple_replace<S, T>::type...>;
  };
}	// namespace detail
    
//! 与えられた型がtupleならばその全要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T=void>
using tuple_replace = typename detail::tuple_replace<S, T>::type;

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
*  tuple_for_each(TUPLES..., FUNC)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return x;
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return std::get<I>(x);
  }
    
  template <class... T>
  struct first_tuple_size;
  template<>
  struct first_tuple_size<>
  {
      constexpr static size_t	value = 0;
  };
  template <class HEAD, class... TAIL>
  struct first_tuple_size<HEAD, TAIL...>
  {
      template <class T_>
      struct tuple_size
      {
	  constexpr static size_t	value = 0;
      };
      template <class... T_>
      struct tuple_size<std::tuple<T_...> >
      {
	  constexpr static size_t	value = sizeof...(T_);
      };

      using TUPLE = std::decay_t<HEAD>;
      
      constexpr static size_t	value = (tuple_size<TUPLE>::value ?
					 tuple_size<TUPLE>::value :
					 first_tuple_size<TAIL...>::value);
  };
    
  template <class FUNC, class... TUPLES> inline void
  tuple_for_each(std::index_sequence<>, FUNC, TUPLES&&...)
  {
  }
  template <size_t I, size_t... IDX, class FUNC, class... TUPLES> inline void
  tuple_for_each(std::index_sequence<I, IDX...>, FUNC f, TUPLES&&... x)
  {
      f(tuple_get<I>(x)...);
      tuple_for_each(std::index_sequence<IDX...>(), f,
		     std::forward<TUPLES>(x)...);
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES>
inline std::enable_if_t<any_tuple<TUPLES...>::value>
tuple_for_each(FUNC f, TUPLES&&... x)
{
    detail::tuple_for_each(std::make_index_sequence<
			       detail::first_tuple_size<TUPLES...>::value>(),
			   f, std::forward<TUPLES>(x)...);
}

/************************************************************************
*  tuple_transform(TUPLES..., FUNC)					*
************************************************************************/
namespace detail
{
  template <class FUNC, class... TUPLES> inline auto
  tuple_transform(std::index_sequence<>, FUNC, TUPLES&&...)
  {
      return std::tuple<>();
  }
  template <class FUNC, class... TUPLES, size_t I, size_t... IDX> inline auto
  tuple_transform(std::index_sequence<I, IDX...>, FUNC f, TUPLES&&... x)
  {
      return std::tuple_cat(std::make_tuple(make_reference_wrapper(
						f(tuple_get<I>(x)...))),
			    tuple_transform(std::index_sequence<IDX...>(),
					    f, std::forward<TUPLES>(x)...));
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any_tuple<TUPLES...>::value>* = nullptr>
inline auto
tuple_transform(FUNC f, TUPLES&&... x)
{
    return detail::tuple_transform(
	       std::make_index_sequence<
		   detail::first_tuple_size<TUPLES...>::value>(),
	       f, std::forward<TUPLES>(x)...);
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> inline auto
select(bool s, const X& x, const Y& y)
{
    return (s ? x : y);
}
    
template <class... S, class X, class Y,
	  std::enable_if_t<any_tuple<X, Y>::value>* = nullptr> inline auto
select(const std::tuple<S...>& s, const X& x, const Y& y)
{
    return tuple_transform([](const auto& t, const auto& u, const auto& v)
			   { return select(t, u, v); }, s, x, y);
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
*  class zip_iterator<ITER_TUPLE, DIFF>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
      template <class ITER_>
      decltype(auto)	operator ()(ITER_ iter)	const	{ return *iter; }
  };
}	// namespace detail
    
template <class ITER_TUPLE, class DIFF=ptrdiff_t>
class zip_iterator : public boost::iterator_facade<
			zip_iterator<ITER_TUPLE>,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			iterator_category<tuple_head<ITER_TUPLE> >,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>()))>
{
  private:
    using super = boost::iterator_facade<
			zip_iterator,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			iterator_category<tuple_head<ITER_TUPLE> >,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>()))>;
    friend	class boost::iterator_core_access;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;
    using stride_t    = tuple_replace<ITER_TUPLE, ptrdiff_t>;
    
  public:
		zip_iterator(ITER_TUPLE iter_tuple)
		    :_iter_tuple(iter_tuple)				{}
    template <class ITER_TUPLE_,
	      std::enable_if_t<
		  std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value>*
	      = nullptr>
		zip_iterator(const zip_iterator<ITER_TUPLE_>& iter)
		    :_iter_tuple(iter.get_iterator_tuple())		{}

    const ITER_TUPLE&
		get_iterator_tuple()	const	{ return _iter_tuple; }
    void	advance(stride_t stride)	{ _iter_tuple += stride; }

  private:
    reference	dereference() const
		{
		    return tuple_transform(detail::generic_dereference(),
					   _iter_tuple);
		}
    template <class ITER_TUPLE_>
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value, bool>
		equal(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			== std::get<0>(_iter_tuple);
		}
    void	increment()
		{
		    ++_iter_tuple;
		}
    void	decrement()
		{
		    --_iter_tuple;
		}
    void	advance(difference_type n)
		{
		    _iter_tuple += n;
		}
    template <class ITER_TUPLE_>
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value,
		     difference_type>
		distance_to(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
		  	 - std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class DIFF=ptrdiff_t, class ITER_TUPLE>
inline zip_iterator<ITER_TUPLE,
		    std::conditional_t<std::is_void<DIFF>::value,
				       tuple_replace<ITER_TUPLE, ptrdiff_t>,
				       DIFF> >
make_zip_iterator(ITER_TUPLE iter_tuple)
{
    return {iter_tuple};
}

/************************************************************************
*  type alias: decayed_iterator_value<ITER>				*
************************************************************************/
namespace detail
{
  template <class ITER>
  struct decayed_iterator_value
  {
      using type = iterator_value<ITER>;
  };
  template <class... ITER>
  struct decayed_iterator_value<zip_iterator<std::tuple<ITER...> > >
  {
      using type = std::tuple<typename decayed_iterator_value<ITER>::type...>;
  };
}	// namespace detail

//! 反復子が指す型を返す．
/*!
  zip_iterator<ITER_TUPLE>::value_type はITER_TUPLE中の各反復子が指す値への
  参照型のtupleであるが，decayed_iterator_value<zip_iterator<ITER_TUPLE> >は
  ITER_TUPLE中の各反復子が指す値そのものの型のtupleを返す．
  \param ITER	反復子
*/
template <class ITER>
using decayed_iterator_value = typename detail::decayed_iterator_value<ITER>
					      ::type;
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
}	// namespace detail
    
template <class TUPLE,
	  enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr> inline auto
begin(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(detail::generic_begin(),
						     t));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return begin(x); }));
}

template <class TUPLE,
	  enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr> inline auto
end(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(detail::generic_end(),
						     t));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return end(x); }));
}

template <class TUPLE,
	  enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr> inline auto
rbegin(TUPLE&& t)
{
    return make_reverse_iterator(end(t));
}

template <class TUPLE,
	  enable_if_t<TU::all_has_begin<TUPLE>::value>* = nullptr> inline auto
rend(TUPLE&& t)
{
    return make_reverse_iterator(begin(t));
}

template <class... T> inline auto
cbegin(const std::tuple<T...>& t)
{
    return begin(t);
}

template <class... T> inline auto
cend(const std::tuple<T...>& t)
{
    return end(t);
}

template <class... T> inline auto
crbegin(const std::tuple<T...>& t)
{
    return rbegin(t);
}

template <class... T> inline auto
crend(const std::tuple<T...>& t)
{
    return rend(t);
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
    return TU::tuple_transform([](const auto& x){ return -x; }, t);
}

template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator +(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x + y; }, l, r);
}

template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator -(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x - y; }, l, r);
}

template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator *(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x * y; }, l, r);
}

template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator /(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x / y; }, l, r);
}

template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator %(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x % y; }, l, r);
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator +=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator -=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator *=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator /=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator %=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T> inline enable_if_t<TU::is_tuple<T>::value, T&>
operator ++(T&& t)
{
    TU::tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T> inline enable_if_t<TU::is_tuple<T>::value, T&>
operator --(T&& t)
{
    TU::tuple_for_each([](auto&& x){ --x; }, t);
    return t;
}

template <class L, class C, class R,
	  enable_if_t<TU::any_tuple<L, C, R>::value>* = nullptr> inline auto
fma(const L& l, const C& c, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y, const auto& z)
			       { return fma(x, y, z); }, l, c, r);
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator &(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x & y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator |(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x | y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator ^(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x ^ y; }, l, r);
}
    
template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator &=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator |=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    TU::tuple_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> inline auto
operator !(const tuple<T...>& t)
{
    return TU::tuple_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator &&(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x && y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator ||(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator ==(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x == y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator !=(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x != y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator <(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x < y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator >(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x > y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator <=(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x <= y; }, l, r);
}
    
template <class L, class R,
	  enable_if_t<TU::any_tuple<L, R>::value>* = nullptr> inline auto
operator >=(const L& l, const R& r)
{
    return TU::tuple_transform([](const auto& x, const auto& y)
			       { return x >= y; }, l, r);
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class... T> inline ostream&
operator <<(ostream& out, const tuple<T...>& t)
{
    out << '(';
    TU::tuple_for_each([&out](const auto& x){ out << ' ' << x; }, t);
    out << ')';

    return out;
}

}	// namespace std
#endif	// !__TU_TUPLE_H
