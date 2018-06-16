/*!
  \file		type_traits.h
  \author	Toshio UESHIBA
  \brief	
*/
#if !defined(TU_TYPE_TRAITS_H)
#define TU_TYPE_TRAITS_H

#include <type_traits>
#include <utility>	// std::declval<T>
#if !defined(NDEBUG)
#  include <boost/core/demangle.hpp>
#endif

namespace TU
{
#if !defined(NDEBUG)
/************************************************************************
*  display type name for debugging					*
************************************************************************/
template <class T> class display_type;

template <class T> inline auto
demangle()
{
    return boost::core::demangle(typeid(T).name());
}
#endif
    
/************************************************************************
*  struct is_convertible<T, C<ARGS...> >				*
************************************************************************/
namespace detail
{
  template <template <class...> class C>
  struct is_convertible
  {
      template <class... ARGS_>
      static std::true_type	check(C<ARGS_...>)			;
      static std::false_type	check(...)				;
  };
}	// namespace detail

template <class T, template <class...> class C>
struct is_convertible
    : decltype(detail::is_convertible<C>::check(std::declval<T>()))	{};
	
/************************************************************************
*  predicate: all<PRED, T...>						*
************************************************************************/
//! 与えられた全ての型が指定された条件を満たすか判定する
/*!
  \param PRED	適用する述語
  \param T...	適用対象となる型の並び
*/
template <template <class> class PRED, class... T>
struct all : std::true_type						{};
template <template <class> class PRED, class S, class... T>
struct all<PRED, S, T...>
    : std::integral_constant<bool,
			     PRED<S>::value && all<PRED, T...>::value>	{};

/************************************************************************
*  predicate: any<PRED, T...>						*
************************************************************************/
//! 与えられた型のうち少なくとも一つが指定された条件を満たすか判定する
/*!
  \param PRED	適用する述語
  \param T...	適用対象となる型の並び
*/
template <template <class> class PRED, class... T>
struct any : std::false_type						{};
template <template <class> class PRED, class S, class... T>
struct any<PRED, S, T...>
    : std::integral_constant<bool,
			     PRED<S>::value || any<PRED, T...>::value>	{};

/************************************************************************
*  void_t<T...>								*
************************************************************************/
namespace detail
{
  template <class... T> struct void_t	{ using type = void; };
}
template <class... T>
using void_t = typename detail::void_t<T...>::type;
    
}	// namespace TU
#endif	// !TU_TYPE_TRAITS_H
