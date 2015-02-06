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
 *  $Id: functional.h 1775 2014-12-24 06:08:59Z ueshiba $
 */
/*!
  \file		tuple.h
  \brief	boost::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <boost/tuple/tuple.hpp>	// boost::tuples::cons
#include <utility>			// std::forward()
#include "TU/functional.h"

namespace boost
{
namespace tuples
{
/************************************************************************
*  struct is_tuple<T>							*
************************************************************************/
template <class T>
class is_tuple
{
  private:
    template <class HEAD_, class TAIL_>
    constexpr static bool	check(cons<HEAD_, TAIL_>*)	{ return true; }
  //constexpr static bool	check(null_type*)		{ return true; }
    constexpr static bool	check(...)			{ return false;}

  public:
    static const bool	value = check(static_cast<T*>(nullptr));
};
    
/************************************************************************
*  tuples::make_cons(HEAD, TAIL)					*
************************************************************************/
template <class HEAD, class TAIL> inline cons<HEAD, TAIL>
make_cons(const HEAD& head, const TAIL& tail)
{
    return cons<HEAD, TAIL>(head, tail);
}
    
/************************************************************************
*  tuples::transform(cons<HEAD, TAIL>, UNARY_FUNC)			*
************************************************************************/
template <class UNARY_FUNC> inline null_type
transform(null_type, UNARY_FUNC)
{
    return null_type();
}
template <class HEAD, class TAIL, class UNARY_FUNC> inline auto
transform(const cons<HEAD, TAIL>& t, const UNARY_FUNC& f)
    -> decltype(make_cons(f(t.get_head()), transform(t.get_tail(), f)))
{
    return make_cons(f(t.get_head()), transform(t.get_tail(), f));
}
    
/************************************************************************
*  tuples::transform(cons<H1, T1>, cons<H2, T2>, BINARY_FUNC)		*
************************************************************************/
template <class BINARY_FUNC> inline null_type
transform(null_type, null_type, BINARY_FUNC)
{
    return null_type();
}
template <class H1, class T1, class H2, class T2, class BINARY_FUNC>
inline auto
transform(const cons<H1, T1>& t1, const cons<H2, T2>& t2, const BINARY_FUNC& f)
    -> decltype(make_cons(f(t1.get_head(), t2.get_head()),
			  transform(t1.get_tail(), t2.get_tail(), f)))
{
    return make_cons(f(t1.get_head(), t2.get_head()),
		     transform(t1.get_tail(), t2.get_tail(), f));
}

/************************************************************************
*  tuples::transform(cons<H1, T1>, cons<H2, T2>,			*
*		     cons<H3, T3>, TRINARY_FUNC)			*
************************************************************************/
template <class TRINARY_FUNC> inline null_type
transform(null_type, null_type, null_type, TRINARY_FUNC)
{
    return null_type();
}
template <class H1, class T1, class H2, class T2,
	  class H3, class T3, class TRINARY_FUNC>
inline auto
transform(const cons<H1, T1>& t1, const cons<H2, T2>& t2,
	  const cons<H3, T3>& t3, const TRINARY_FUNC& f)
    -> decltype(make_cons(f(t1.get_head(), t2.get_head(), t3.get_head()),
			  transform(t1.get_tail(),
				    t2.get_tail(), t3.get_tail(), f)))
{
    return make_cons(f(t1.get_head(), t2.get_head(), t3.get_head()),
		     transform(t1.get_tail(), t2.get_tail(), t3.get_tail(), f));
}

/************************************************************************
*  tuples::for_each(cons<HEAD, TAIL>, UNARY_FUNC)			*
************************************************************************/
template <class UNARY_FUNC> inline void
for_each(null_type, UNARY_FUNC)
{
}
template <class HEAD, class TAIL, class UNARY_FUNC> inline void
for_each(cons<HEAD, TAIL>& t, const UNARY_FUNC& f)
{
    f(t.get_head());
    for_each(t.get_tail(), f);
}

/************************************************************************
*  tuples::for_each(cons<H1, T1>, cons<H2, T2>, UNARY_FUNC)		*
************************************************************************/
template <class BINARY_FUNC> inline void
for_each(null_type, null_type, BINARY_FUNC)
{
}
template <class H1, class T1, class H2, class T2, class BINARY_FUNC> inline void
for_each(const cons<H1, T1>& t1, cons<H2, T2>& t2, const BINARY_FUNC& f)
{
    f(t1.get_head(), t2.get_head());
    for_each(t1.get_tail(), t2.get_tail(), f);
}
    
/************************************************************************
*  tuples::tuple_cat(cons<H1, T1>, T)					*
************************************************************************/
template <class T> inline const T&
tuple_cat(null_type, const T& x)
{
    return x;
}
template <class HEAD, class TAIL, class T> inline auto
tuple_cat(const cons<HEAD, TAIL>& t, const T& x)
    -> decltype(make_cons(t.get_head(), tuple_cat(t.get_tail(), x)))
{
    return make_cons(t.get_head(), tuple_cat(t.get_tail(), x));
}
    
/************************************************************************
*  struct generic_function<FUNC>					*
************************************************************************/
template <template <class> class FUNC>
struct generic_function
{
    template <class T_> auto
    operator ()(const T_& x) const -> decltype(FUNC<T_>()(x))
    {
	return FUNC<T_>()(x);
    }

    template <class T_> auto
    operator ()(const T_& x, const T_& y) const -> decltype(FUNC<T_>()(x, y))
    {
	return FUNC<T_>()(x, y);
    }
};

/************************************************************************
*  struct generic_binary_function<FUNC>					*
************************************************************************/
template <template <class, class> class FUNC>
struct generic_binary_function
{
    template <class S_, class T_> auto
    operator ()(const S_& x, const T_& y) const -> decltype(FUNC<S_, T_>()(x, y))
    {
	return FUNC<S_, T_>()(x, y);
    }
};

/************************************************************************
*  struct generic_assgin<ASSIGN>					*
************************************************************************/
template <template <class, class> class ASSIGN>
struct generic_assign
{
    template <class S_, class T_> T_&
    operator ()(const S_& x, T_&& y) const
    {
	return ASSIGN<S_, T_>()(x, std::forward<T_>(y));
    }
};

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class HEAD, class TAIL> inline auto
operator -(const cons<HEAD, TAIL>& x)
    -> decltype(transform(x, generic_function<std::negate>()))
{
    return transform(x, generic_function<std::negate>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator +(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_binary_function<TU::plus>()))
{
    return transform(x, y, generic_binary_function<TU::plus>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator -(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_binary_function<TU::minus>()))
{
    return transform(x, y, generic_binary_function<TU::minus>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator *(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_binary_function<TU::multiplies>()))
{
    return transform(x, y, generic_binary_function<TU::multiplies>());
}

template <class T, class HEAD, class TAIL,
	  class=typename std::enable_if<!is_tuple<T>::value>::type> inline auto
operator *(const T& c, const cons<HEAD, TAIL>& x)
    -> decltype(transform(x, std::bind(generic_binary_function<TU::multiplies>(),
				       c, std::placeholders::_1)))
{
    return transform(x, std::bind(generic_binary_function<TU::multiplies>(),
				  c, std::placeholders::_1));
}

template <class HEAD, class TAIL, class T,
	  class=typename std::enable_if<!is_tuple<T>::value>::type> inline auto
operator *(const cons<HEAD, TAIL>& x, const T& c)
    -> decltype(transform(x, std::bind(generic_binary_function<TU::multiplies>(),
				       std::placeholders::_1, c)))
{
    return transform(x, std::bind(generic_binary_function<TU::multiplies>(),
				  std::placeholders::_1, c));
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator /(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_binary_function<TU::divides>()))
{
    return transform(x, y, generic_binary_function<TU::divides>());
}
    
template <class HEAD, class TAIL, class T,
	  class=typename std::enable_if<!is_tuple<T>::value>::type> inline auto
operator /(const cons<HEAD, TAIL>& x, const T& c)
    -> decltype(transform(x, std::bind(generic_binary_function<TU::divides>(),
				       std::placeholders::_1, c)))
{
    return transform(x, std::bind(generic_binary_function<TU::divides>(),
				  std::placeholders::_1, c));
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator %(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::modulus>()))
{
    return transform(x, y, generic_function<std::modulus>());
}

template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator +=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::plus_assign>());
    return y;
}

template <class L, class HEAD, class TAIL,
	  class=typename std::enable_if<
	      is_tuple<typename std::decay<L>::type>::value>::type> inline L&
operator -=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::minus_assign>());
    return y;
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator *=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::multiplies_assign>());
    return y;
}
    
template <class HEAD, class TAIL, class T>
inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
operator *=(cons<HEAD, TAIL>& y, const T& c)
{
    for_each(y, std::bind(generic_assign<TU::multiplies_assign>(),
			  c, std::placeholders::_1));
    return y;
}
    
template <class HEAD, class TAIL, class T>
inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
operator *=(cons<HEAD, TAIL>&& y, const T& c)
{
    for_each(y, std::bind(generic_assign<TU::multiplies_assign>(),
			  c, std::placeholders::_1));
    return y;
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator /=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::divides_assign>());
    return y;
}
    
template <class HEAD, class TAIL, class T>
inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
operator /=(cons<HEAD, TAIL>& y, const T& c)
{
    for_each(y, std::bind(generic_assign<TU::divides_assign>(),
			  c, std::placeholders::_1));
    return y;
}
    
template <class HEAD, class TAIL, class T>
inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
operator /=(cons<HEAD, TAIL>&& y, const T& c)
{
    for_each(y, std::bind(generic_assign<TU::divides_assign>(),
			  c, std::placeholders::_1));
    return y;
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator %=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::modulus_assign>());
    return y;
}
    
/************************************************************************
*  Bit operators							*
************************************************************************/
template <class H1, class T1, class H2, class T2> inline auto
operator &(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::bit_and>()))
{
    return transform(x, y, generic_function<std::bit_and>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator |(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::bit_or>()))
{
    return transform(x, y, generic_function<std::bit_or>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator ^(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::bit_xor>()))
{
    return transform(x, y, generic_function<std::bit_xor>());
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator &=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::bit_and_assign>());
    return y;
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator |=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::bit_or_assign>());
    return y;
}
    
template <class L, class HEAD, class TAIL>
inline typename std::enable_if<is_tuple<typename std::decay<L>::type>::value,
			       L&>::type
operator ^=(L&& y, const cons<HEAD, TAIL>& x)
{
    for_each(x, y, generic_assign<TU::bit_xor_assign>());
    return y;
}
    
/************************************************************************
*  Logical operators							*
************************************************************************/
template <class HEAD, class TAIL> inline auto
operator !(const cons<HEAD, TAIL>& x)
    -> decltype(transform(x, generic_function<std::logical_not>()))
{
    return transform(x, generic_function<std::logical_not>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator &&(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::logical_and>()))
{
    return transform(x, y, generic_function<std::logical_and>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator ||(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::logical_or>()))
{
    return transform(x, y, generic_function<std::logical_or>());
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class H1, class T1, class H2, class T2> inline auto
operator ==(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::equal_to>()))
{
    return transform(x, y, generic_function<std::equal_to>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator !=(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::not_equal_to>()))
{
    return transform(x, y, generic_function<std::not_equal_to>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator <(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::less>()))
{
    return transform(x, y, generic_function<std::less>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator >(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::greater>()))
{
    return transform(x, y, generic_function<std::greater>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator <=(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::less_equal>()))
{
    return transform(x, y, generic_function<std::less_equal>());
}
    
template <class H1, class T1, class H2, class T2> inline auto
operator >=(const cons<H1, T1>& x, const cons<H2, T2>& y)
    -> decltype(transform(x, y, generic_function<std::greater_equal>()))
{
    return transform(x, y, generic_function<std::greater_equal>());
}

/************************************************************************
*  struct htuple<T, N>							*
************************************************************************/
template <class T, size_t N>
struct htuple
{
    typedef cons<T, typename htuple<T, N-1>::type>	type;

    static type	make(const T& x)
		{
		    T	y = x;
		    return type(x, htuple<T, N-1>::make(++y));
		}
};
template <class T>
struct htuple<T, 0>
{
    typedef null_type					type;

    static type	make(const T&)				{ return type(); }
};

}	// End of namespace boost::tuples

/************************************************************************
*  typedef of htuple<T, N>						*
************************************************************************/
template <class T, size_t N=1>
using	htuple = typename tuples::htuple<T, N>::type;

template <size_t N, class T> htuple<T, N>
make_contiguous_htuple(const T& x)
{
    return tuples::htuple<T, N>::make(x);
}
    
}	// End of namespace boost

namespace TU
{
namespace detail
{
  template <class T>
  struct tuple_head
  {
      typedef T		type;
  };
  template <class HEAD, class TAIL>
  struct tuple_head<boost::tuples::cons<HEAD, TAIL> >
  {
      typedef HEAD	type;
  };
  template <BOOST_PP_ENUM_PARAMS(10, class T)>
  struct tuple_head<boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)> >
      : tuple_head<typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)>::inherited>
  {
  };
}	// End of namespace TU::detail

/************************************************************************
*  struct tuple_head<T>							*
************************************************************************/
//! 与えられた型がtupleまたはconsならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using	tuple_head = typename detail::tuple_head<T>::type;
    
/************************************************************************
*  Selection								*
************************************************************************/
template <class T> inline const T&
select(bool s, const T& x, const T& y)
{
    return (s ? x : y);
}
    
struct generic_select
{
    template <class S_, class T_> const T_&
    operator ()(const S_& s, const T_& x, const T_& y) const
    {
	return select(s, x, y);
    }
};

template <class H1, class T1, class H2, class T2, class H3, class T3>
inline auto
select(const boost::tuples::cons<H1, T1>& s,
       const boost::tuples::cons<H2, T2>& x, const boost::tuples::cons<H3, T3>& y)
    -> decltype(boost::tuples::transform(s, x, y, generic_select()))
{
    return boost::tuples::transform(s, x, y, generic_select());
}

/************************************************************************
*  class unarizer<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって多変数関数を1変数関数に変換
template <class FUNC>
class unarizer
{
  public:
    typedef FUNC			functor_type;
    typedef typename FUNC::result_type	result_type;

  public:
    unarizer(const FUNC& func=FUNC())	:_func(func)	{}

    result_type	operator ()(boost::tuples::null_type) const
		{
		    return _func();
		}
    template <class HEAD, class TAIL>
    result_type	operator ()(const boost::tuples::cons<HEAD, TAIL>& arg) const
		{
		    return exec(arg,
				std::integral_constant<
				    size_t,
				    1 + boost::tuples::length<TAIL>::value>());
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    template <class TUPLE>
    result_type	exec(const TUPLE& arg, std::integral_constant<size_t, 1>) const
		{
		    return _func(boost::get<0>(arg));
		}
    template <class TUPLE>
    result_type	exec(const TUPLE& arg, std::integral_constant<size_t, 2>) const
		{
		    return _func(boost::get<0>(arg), boost::get<1>(arg));
		}
    template <class TUPLE>
    result_type	exec(const TUPLE& arg, std::integral_constant<size_t, 3>) const
		{
		    return _func(boost::get<0>(arg),
				 boost::get<1>(arg), boost::get<2>(arg));
		}
    template <class TUPLE>
    result_type	exec(const TUPLE& arg, std::integral_constant<size_t, 4>) const
		{
		    return _func(boost::get<0>(arg), boost::get<1>(arg),
				 boost::get<2>(arg), boost::get<3>(arg));
		}

  private:
    const FUNC&	_func;
};

template <class FUNC> inline unarizer<FUNC>
make_unarizer(const FUNC& func)
{
    return unarizer<FUNC>(func);
}

}	// End of namespace TU

#endif	// !__TU_TUPLE_H
