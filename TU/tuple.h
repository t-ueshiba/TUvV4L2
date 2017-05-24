/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef TU_TUPLE_H
#define TU_TUPLE_H

#include <tuple>
#include <utility>		// for std::index_sequence<IDX...>
#include <type_traits>		// for std::enable_if_t<B, T>
#include <iostream>

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
*  predicate: is_tuple<T>						*
************************************************************************/
namespace detail
{
  template <class T>
  struct check_tuple : std::false_type					{};
  template <class... T>
  struct check_tuple<std::tuple<T...> > : std::true_type		{};
}	// namespace detail

//! 与えられた型が std::tuple 又はそれへの参照であるか判定する
/*!
  T が std::tuple に変換可能でも，std::tuple そのもの，又はそれへの参照で
  なければ false
  \param T	判定対象となる型
*/ 
template <class T>
using is_tuple		= detail::check_tuple<std::decay_t<T> >;

/************************************************************************
*  predicate: any_tuple<ARGS...>					*
************************************************************************/
//! 少なくとも1つのテンプレート引数が std::tuple 又はそれへの参照であるか判定する
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
*  Arithmetic operators							*
************************************************************************/
template <class... T> inline auto
operator -(const std::tuple<T...>& t)
{
    return tuple_transform([](const auto& x){ return -x; }, t);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator +(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x + y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator -(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x - y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator *(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x * y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator /(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x / y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator %(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x % y; }, l, r);
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
	  std::enable_if_t<any_tuple<L, C, R>::value>* = nullptr> inline auto
fma(const L& l, const C& c, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y, const auto& z)
			   { return fma(x, y, z); }, l, c, r);
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator &(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x & y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator |(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x | y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator ^(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x ^ y; }, l, r);
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
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator &&(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x && y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator ||(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator ==(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x == y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator !=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x != y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator <(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x < y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator >(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x > y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator <=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x <= y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any_tuple<L, R>::value>* = nullptr> inline auto
operator >=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x >= y; }, l, r);
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
