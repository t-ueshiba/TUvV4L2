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
 *  $Id: functional.h 2200 2017-01-25 00:51:35Z ueshiba $
 */
/*!
  \file		functional.h
  \brief	関数オブジェクトの定義と実装
*/
#ifndef __TU_FUNCTIONAL_H
#define __TU_FUNCTIONAL_H

#include <cstddef>				// size_t
#include <cmath>				// std::sqrt()
#include <functional>				// std::bind()
#include <type_traits>				// std::integral_constant
#include <numeric>				// std::accumulate()

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
      template <class... ARGS>
      static std::true_type	check(C<ARGS...>)			;
      static std::false_type	check(...)				;
  };
}	// namespace detail

template <class T, template <class...> class C>
struct is_convertible
    : decltype(detail::is_convertible<C>::check(std::declval<T>()))	{};
	
/************************************************************************
*  struct generic_function<FUNC>					*
************************************************************************/
//! テンプレート引数を1つ持つ関数オブジェクトをgenericにするアダプタ
template <template <class> class FUNC>
struct generic_function
{
    template <class T_>
    auto	operator ()(T_&& x) const
		{
		    return FUNC<T_>()(std::forward<T_>(x));
		}

    template <class T_>
    auto	operator ()(const T_& x, const T_& y) const
		{
		    return FUNC<T_>()(x, y);
		}
};

/************************************************************************
*  struct negate							*
************************************************************************/
//! 符号反転
struct negate
{
    template <class T_>
    auto	operator ()(const T_& x) const
		{
		    return -x;
		}
};
    
/************************************************************************
*  struct generic_binary_function<FUNC>					*
************************************************************************/
//! テンプレート引数を2つ持つ関数オブジェクトをgenericにするアダプタ
template <template <class, class> class FUNC>
struct generic_binary_function
{
    template <class S_, class T_>
    auto	operator ()(S_&& x, T_&& y) const
		{
		    return FUNC<S_, T_>()(std::forward<S_>(x),
					  std::forward<T_>(y));
		}
};

/************************************************************************
*  struct plus								*
************************************************************************/
//! 加算
struct plus
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x + y;
		}
};
    
/************************************************************************
*  struct minus								*
************************************************************************/
//! 減算
struct minus
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x - y;
		}
};
    
/************************************************************************
*  struct multiplies							*
************************************************************************/
//! 乗算
struct multiplies
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x * y;
		}
};
    
/************************************************************************
*  struct divides							*
************************************************************************/
//! 除算
struct divides
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x / y;
		}
};
    
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

/************************************************************************
*  struct equal_to							*
************************************************************************/
//! 等しい
struct equal_to
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x == y;
		}
};
    
/************************************************************************
*  struct not_equal_to							*
************************************************************************/
//! 等しくない
struct not_equal_to
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x != y;
		}
};
    
/************************************************************************
*  struct less								*
************************************************************************/
//! より小さい
struct less
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x < y;
		}
};
    
/************************************************************************
*  struct greater							*
************************************************************************/
//! より大きい
struct greater
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x > y;
		}
};
    
/************************************************************************
*  struct less_equal							*
************************************************************************/
//! より小さいか等しい
struct less_equal
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x <= y;
		}
};
    
/************************************************************************
*  struct greater_equal							*
************************************************************************/
//! より大きいか等しい
struct greater_equal
{
    template <class S_, class T_>
    auto	operator ()(const S_& x, const T_& y) const
		{
		    return x >= y;
		}
};
    
/************************************************************************
*  Selection								*
************************************************************************/
template <class S, class T> inline auto
select(bool s, const S& x, const T& y)
{
    return (s ? x : y);
}
    
/************************************************************************
*  struct generic_select						*
************************************************************************/
struct generic_select
{
    template <class R_, class S_, class T_> auto
    operator ()(const R_& s, const S_& x, const T_& y) const
    {
	return select(s, x, y);
    }
};


}	// End of namespace TU
#endif	// !__TU_FUNCTIONAL_H
