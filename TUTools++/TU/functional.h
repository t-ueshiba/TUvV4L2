/*!
  \file		functional.h
  \brief	関数オブジェクトの定義と実装
*/
#ifndef __TU_FUNCTIONAL_H
#define __TU_FUNCTIONAL_H

#include <type_traits>				// std::integral_constant

namespace TU
{
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
*  struct identity							*
************************************************************************/
//! 恒等関数
struct identity
{
    template <class T_>
    T_&	operator ()(T_&& x)			const	{ return x; }
};

/************************************************************************
*  struct assign							*
************************************************************************/
//! 代入
struct assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y = x; return y; }
};

/************************************************************************
*  struct plus_assign							*
************************************************************************/
//! 引数を加算
struct plus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y += x; return y; }
};

/************************************************************************
*  struct minus_assign							*
************************************************************************/
//! 引数を減算
struct minus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y -= x; return y; }
};

/************************************************************************
*  struct multiplies_assign						*
************************************************************************/
//! 引数を乗算
struct multiplies_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y *= x; return y; }
};

/************************************************************************
*  struct divides_assign						*
************************************************************************/
//! 引数を除算
struct divides_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y /= x; return y; }
};

/************************************************************************
*  struct modulus_assign						*
************************************************************************/
//! 引数で割った時の剰余を代入
struct modulus_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y %= x; return y; }
};

/************************************************************************
*  struct bit_and_assign						*
************************************************************************/
//! 引数とのAND
struct bit_and_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y &= x; return y; }
};

/************************************************************************
*  struct bit_or_assign							*
************************************************************************/
//! 引数とのOR
struct bit_or_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y |= x; return y; }
};

/************************************************************************
*  struct bit_xor_assign						*
************************************************************************/
//! 引数とのXOR
struct bit_xor_assign
{
    template <class T_, class S_>
    T_&	operator ()(T_&& y, const S_& x)	const	{ y ^= x; return y; }
};

}	// End of namespace TU
#endif	// !__TU_FUNCTIONAL_H
