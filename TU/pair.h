/*!
  \file		pair.h
  \author	Toshio UESHIBA
  \brief	std::pairの用途拡張のためのユティリティ
*/
#ifndef __TU_PAIR_H
#define __TU_PAIR_H

#include <utility>
#include <iostream>

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

//! 与えられた型が std::pair 又はそれへの参照であるか判定する
/*!
  T が std::pair に変換可能でも，std::pair そのもの，又はそれへの参照で
  なければ false
  \param T	判定対象となる型
*/ 
template <class T>
using is_pair = detail::check_pair<std::decay_t<T> >;
    
/************************************************************************
*  predicate: any_pair<ARGS...>						*
************************************************************************/
//! 少なくとも1つのテンプレート引数が std::pair 又はそれへの参照であるか判定する
/*!
  \param ARGS...	判定対象となる型の並び
*/
template <class... ARGS>
struct any_pair;
template <>
struct any_pair<> : std::false_type					{};
template <class ARG, class... ARGS>
struct any_pair<ARG, ARGS...>
    : std::integral_constant<bool, (is_pair<ARG>::value ||
				    any_pair<ARGS...>::value)>		{};
    
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
    
/************************************************************************
*  pair_for_each(PAIRS..., FUNC)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_pair<T>::value>* = nullptr>
  inline decltype(auto)
  pair_get(T&& x)
  {
      return x;
  }
  template <size_t I, class T, std::enable_if_t<is_pair<T>::value>* = nullptr>
  inline decltype(auto)
  pair_get(T&& x)
  {
      return std::get<I>(x);
  }
}	// namespace detail
    
template <class FUNC, class... PAIRS>
inline std::enable_if_t<any_pair<PAIRS...>::value>
pair_for_each(FUNC f, PAIRS&&... x)
{
    f(detail::pair_get<0>(x)...);
    f(detail::pair_get<1>(x)...);
}

/************************************************************************
*  pair_transform(PAIRS..., FUNC)					*
************************************************************************/
template <class FUNC, class... PAIRS,
	  std::enable_if_t<any_pair<PAIRS...>::value>* = nullptr> inline auto
pair_transform(FUNC f, PAIRS&&... x)
{
    return std::make_pair(f(detail::pair_get<0>(x)...),
			  f(detail::pair_get<1>(x)...));
}

}	// namespace TU

namespace std
{
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class S, class T> inline auto
operator -(const pair<S, T>& t)
{
    return TU::pair_transform([](const auto& x){ return -x; }, t);
}

template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator +(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x + y; }, l, r);
}

template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator -(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x - y; }, l, r);
}

template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x * y; }, l, r);
}

template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator /(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x / y; }, l, r);
}

template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator %(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x % y; }, l, r);
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator +=(L&& l, const R& r)
{
    TU::pair_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator -=(L&& l, const R& r)
{
    TU::pair_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator *=(L&& l, const R& r)
{
    TU::pair_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator /=(L&& l, const R& r)
{
    TU::pair_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator %=(L&& l, const R& r)
{
    TU::pair_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T> inline enable_if_t<TU::is_pair<T>::value, T&>
operator ++(T&& t)
{
    TU::pair_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T> inline enable_if_t<TU::is_pair<T>::value, T&>
operator --(T&& t)
{
    TU::pair_for_each([](auto&& x){ --x; }, t);
    return t;
}

template <class L, class C, class R,
	  enable_if_t<TU::any_pair<L, C, R>::value>* = nullptr> inline auto
fma(const L& l, const C& c, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y, const auto& z)
			       { return fma(x, y, z); }, l, c, r);
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator &(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x & y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator |(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x | y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator ^(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x ^ y; }, l, r);
}
    
template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator &=(L&& l, const R& r)
{
    TU::pair_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator |=(L&& l, const R& r)
{
    TU::pair_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R> inline enable_if_t<TU::is_pair<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    TU::pair_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class S, class T> inline auto
operator !(const pair<S, T>& t)
{
    return TU::pair_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator &&(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x && y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator ||(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator ==(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x == y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator !=(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x != y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator <(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x < y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator >(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x > y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator <=(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x <= y; }, l, r);
}
    
template <class L, class R, enable_if_t<TU::any_pair<L, R>::value>* = nullptr>
inline auto
operator >=(const L& l, const R& r)
{
    return TU::pair_transform([](const auto& x, const auto& y)
			      { return x >= y; }, l, r);
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class S, class T> inline ostream&
operator <<(ostream& out, const std::pair<S, T>& x)
{
    return out << '[' << x.first << ' ' << x.second << ']';
}
    
}	// namespace std

#endif	// !__TU_PAIR_H
