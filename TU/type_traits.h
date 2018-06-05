/*!
  \file		type_traits.h
  \author	Toshio UESHIBA
  \brief	
*/
#if !defined(TU_TYPE_TRAITS_H)
#define TU_TYPE_TRAITS_H

#include <type_traits>
#include <utility>	// std::declval<T>

namespace TU
{
/************************************************************************
*  display type for debugging purpose					*
************************************************************************/
template <class T> class display_type;
    
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
struct all
{
    constexpr static bool	value = true;
};
template <template <class> class PRED, class S, class... T>
struct all<PRED, S, T...>
{
    constexpr static bool	value = PRED<S>::value &&
					all<PRED, T...>::value;
};

/************************************************************************
*  predicate: any<PRED, T...>						*
************************************************************************/
//! 与えられた型のうち少なくとも一つが指定された条件を満たすか判定する
/*!
  \param PRED	適用する述語
  \param T...	適用対象となる型の並び
*/
template <template <class> class PRED, class... T>
struct any
{
    constexpr static bool	value = false;
};
template <template <class> class PRED, class S, class... T>
struct any<PRED, S, T...>
{
    constexpr static bool	value = PRED<S>::value ||
					any<PRED, T...>::value;
};

}	// namespace TU
#endif	// !TU_TYPE_TRAITS_H
