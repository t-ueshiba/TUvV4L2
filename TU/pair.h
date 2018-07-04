/*!
  \file		pair.h
  \author	Toshio UESHIBA
  \brief	std::pairの用途拡張のためのユティリティ
*/
#ifndef TU_PAIR_H
#define TU_PAIR_H

#include <iostream>
#include "TU/type_traits.h"	// for TU::any<PRED, T...>

namespace TU
{
/************************************************************************
*  predicate is_pair<T>							*
************************************************************************/
namespace detail
{
  template <class T>
  struct check_pair : std::false_type					{};
  template <class S, class T>
  struct check_pair<std::pair<S, T> > : std::true_type			{};
}	// namespace detail
    
/*!
  T が std::pair に変換可能でも，std::pair そのもの，またはそれへの参照
  でなければ false（std::tuple に対して true にならないための措置）
  \param T	判定対象となる型
*/ 
template <class T>
using is_pair = detail::check_pair<std::decay_t<T> >;
    
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
    using rightmost_type	= typename pair_traits<T>::rightmost_type;
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
    
/************************************************************************
*  pair_for_each(FUNC, PAIRS&&...)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_pair<T>::value>* = nullptr>
  inline decltype(auto)
  pair_get(T&& x)
  {
      return std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_pair<T>::value>* = nullptr>
  inline decltype(auto)
  pair_get(T&& x)
  {
      return std::get<I>(std::forward<T>(x));
  }
}	// namespace detail
    
template <class FUNC, class... PAIRS>
inline std::enable_if_t<any<is_pair, PAIRS...>::value>
pair_for_each(FUNC f, PAIRS&&... x)
{
    f(detail::pair_get<0>(std::forward<PAIRS>(x))...);
    f(detail::pair_get<1>(std::forward<PAIRS>(x))...);
}

/************************************************************************
*  pair_transform(FUNC, PAIRS&&...)					*
************************************************************************/
template <class FUNC, class... PAIRS,
	  std::enable_if_t<any<is_pair, PAIRS...>::value>* = nullptr>
inline auto
pair_transform(FUNC f, PAIRS&&... x)
{
    return std::make_pair(f(detail::pair_get<0>(std::forward<PAIRS>(x))...),
			  f(detail::pair_get<1>(std::forward<PAIRS>(x))...));
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class E, std::enable_if_t<is_pair<E>::value>* = nullptr> inline auto
operator -(E&& expr)
{
    return pair_transform([](auto&& x){ return -std::forward<decltype(x)>(x); },
			  std::forward<E>(expr));
}

template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator +(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 + std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator -(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 - std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator *(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 * std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator /(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 / std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator %(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 % std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator +=(L&& l, const R& r)
{
    pair_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator -=(L&& l, const R& r)
{
    pair_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator *=(L&& l, const R& r)
{
    pair_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator /=(L&& l, const R& r)
{
    pair_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator %=(L&& l, const R& r)
{
    pair_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T> inline std::enable_if_t<is_pair<T>::value, T&>
operator ++(T&& t)
{
    pair_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T> inline std::enable_if_t<is_pair<T>::value, T&>
operator --(T&& t)
{
    pair_for_each([](auto&& x){ --x; }, t);
    return t;
}

template <class L, class C, class R,
	  std::enable_if_t<any<is_pair, L, C, R>::value>* = nullptr>
inline auto
fma(L&& l, C&& c, R&& r)
{
    return pair_transform([](auto&& x, auto&& y, auto&& z)
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
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator &(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 & std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator |(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 | std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator ^(L&& l, R&& r)
{
    return pair_transform([](auto&& x, auto&& y)
			  { return std::forward<decltype(x)>(x)
				 ^ std::forward<decltype(y)>(y); },
			  std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator &=(L&& l, const R& r)
{
    pair_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator |=(L&& l, const R& r)
{
    pair_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R> inline std::enable_if_t<is_pair<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    pair_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class S, class T> inline auto
operator !(const std::pair<S, T>& t)
{
    return pair_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator &&(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x && y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator ||(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator ==(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x == y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator !=(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x != y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator <(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x < y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator >(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x > y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator <=(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x <= y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_pair, L, R>::value>* = nullptr> inline auto
operator >=(const L& l, const R& r)
{
    return pair_transform([](const auto& x, const auto& y)
			  { return x >= y; }, l, r);
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class S, class T> inline std::ostream&
operator <<(std::ostream& out, const std::pair<S, T>& x)
{
    return out << '[' << x.first << ' ' << x.second << ']';
}

}	// namespace TU
#endif	// !TU_PAIR_H
