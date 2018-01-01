/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef TU_TUPLE_H
#define TU_TUPLE_H

#include <tuple>
#include <iostream>
#include "TU/type_traits.h"	// for TU::any<PRED, T...>

namespace TU
{
/************************************************************************
*  predicate: is_tuple<T>						*
************************************************************************/
namespace detail
{
  template <class... T>
  std::true_type	check_tuple(std::tuple<T...>)			;
  std::false_type	check_tuple(...)				;
}	// namespace detail

//! 与えられた型が std::tuple 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/ 
template <class T>
using is_tuple = decltype(detail::check_tuple(std::declval<T>()));

/************************************************************************
*  type alias: tuple_head<T>						*
************************************************************************/
namespace detail
{
  template <class HEAD, class... TAIL>
  HEAD	tuple_head(std::tuple<HEAD, TAIL...>)				;
  template <class T>
  T	tuple_head(T)							;
}	// namespace detail
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = decltype(detail::tuple_head(std::declval<T>()));

/************************************************************************
*  type alias: replace_element<S, T>					*
************************************************************************/
namespace detail
{
  template <class S, class T>
  struct replace_element : std::conditional<std::is_void<T>::value, S, T>
  {
  };
  template <class... S, class T>
  struct replace_element<std::tuple<S...>, T>
  {
      using type = std::tuple<typename replace_element<S, T>::type...>;
  };
}	// namespace detail
    
//! 与えられた型がstd::tupleならばその要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T>
using replace_element = typename detail::replace_element<S, T>::type;

/************************************************************************
*  make_reference_wrapper(T&&)						*
************************************************************************/
//! 与えられた値の型に応じて実引数を生成する
/*!
  \param x	関数に渡す引数
  \return	xが右辺値参照ならばx自身，定数参照ならばstd::cref(x),
		非定数参照ならばstd::ref(x)
*/
template <class T>
inline std::conditional_t<std::is_lvalue_reference<T>::value,
			  std::reference_wrapper<std::remove_reference_t<T> >,
			  T&&>
make_reference_wrapper(T&& x)
{
    return std::forward<T>(x);
}
    
/************************************************************************
*  tuple_for_each(FUNC, TUPLES&&...)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return std::get<I>(std::forward<T>(x));
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
      f(tuple_get<I>(std::forward<TUPLES>(x))...);
      tuple_for_each(std::index_sequence<IDX...>(), f,
		     std::forward<TUPLES>(x)...);
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES>
inline std::enable_if_t<any<is_tuple, TUPLES...>::value>
tuple_for_each(FUNC f, TUPLES&&... x)
{
    detail::tuple_for_each(std::make_index_sequence<
			       detail::first_tuple_size<TUPLES...>::value>(),
			   f, std::forward<TUPLES>(x)...);
}

/************************************************************************
*  tuple_transform(FUNC, TUPLES&&...)					*
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
      return std::tuple_cat(
		std::make_tuple(
		    make_reference_wrapper(
			f(tuple_get<I>(std::forward<TUPLES>(x))...))),
		tuple_transform(std::index_sequence<IDX...>(),
				f, std::forward<TUPLES>(x)...));
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_tuple, TUPLES...>::value>* = nullptr>
inline auto
tuple_transform(FUNC f, TUPLES&&... x)
{
    return detail::tuple_transform(
	       std::make_index_sequence<
		   detail::first_tuple_size<TUPLES...>::value>(),
	       f, std::forward<TUPLES>(x)...);
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class E,
	  std::enable_if_t<is_tuple<E>::value>* = nullptr> inline auto
operator -(E&& expr)
{
    return tuple_transform([](auto&& x)
			   { return -std::forward<decltype(x)>(x); },
			   std::forward<E>(expr));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator +(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  + std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator -(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  - std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator *(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  * std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator /(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  / std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator %(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  % std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator +=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator -=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator *=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator /=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator %=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T> inline std::enable_if_t<is_tuple<T>::value, T&>
operator ++(T&& t)
{
    tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T> inline std::enable_if_t<is_tuple<T>::value, T&>
operator --(T&& t)
{
    tuple_for_each([](auto&& x){ --x; }, t);
    return t;
}

template <class L, class C, class R,
	  std::enable_if_t<any<is_tuple, L, C, R>::value>* = nullptr>
inline auto
fma(L&& l, C&& c, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y, auto&& z)
			   { return fma(std::forward<decltype(x)>(x),
					std::forward<decltype(y)>(y),
					std::forward<decltype(z)>(z)); },
			   std::forward<L>(l),
			   std::forward<C>(c),
			   std::forward<R>(r));
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator &(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  & std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator |(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  | std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator ^(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  ^ std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator &=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator |=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_tuple<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> inline auto
operator !(const std::tuple<T...>& t)
{
    return tuple_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator &&(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x && y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator ||(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator ==(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x == y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator !=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x != y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator <(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x < y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator >(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x > y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator <=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x <= y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator >=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x >= y; }, l, r);
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> inline auto
select(bool s, X&& x, Y&& y)
{
    return (s ? std::forward<X>(x) : std::forward<Y>(y));
}
    
template <class... S, class X, class Y,
	  std::enable_if_t<any<is_tuple, X, Y>::value>* = nullptr> inline auto
select(const std::tuple<S...>& s, X&& x, Y&& y)
{
    return tuple_transform([](const auto& t, auto&& u, auto&& v)
			   { return select(t,
					   std::forward<decltype(u)>(u),
					   std::forward<decltype(v)>(v)); },
			   s, std::forward<X>(x), std::forward<Y>(y));
}

}	// namespace TU

namespace std
{
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
#endif	// !TU_TUPLE_H
