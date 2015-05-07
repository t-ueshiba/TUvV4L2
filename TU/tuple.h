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

#include <boost/tuple/tuple.hpp>
#include <utility>			// std::forward()
#include "TU/functional.h"

namespace boost
{
namespace tuples
{
  /**********************************************************************
  *  struct is_tuple<T>							*
  **********************************************************************/
  template <class HEAD, class TAIL>
  static true_type	tuple_check(cons<HEAD, TAIL>)		;
  static false_type	tuple_check(...)			;
    
  template <class T>
  struct is_tuple : decltype(tuple_check(std::declval<T>()))	{};

  /**********************************************************************
  *  make_contiguous_htuple(T, TU::index_sequence<IDX...>)		*
  **********************************************************************/
  template <class T, size_t ...IDX> static inline auto
  make_contiguous_htuple(T&& x, TU::index_sequence<IDX...>)
      -> decltype(boost::make_tuple((x + IDX)...))
  {
      return boost::make_tuple((x + IDX)...);
  }

  /**********************************************************************
  *  cons_for_each(cons<HEAD, TAIL>, UNARY_FUNC)			*
  **********************************************************************/
  template <class UNARY_FUNC> inline void
  cons_for_each(null_type, UNARY_FUNC)
  {
  }
  template <class HEAD, class TAIL, class UNARY_FUNC> inline void
  cons_for_each(cons<HEAD, TAIL>& x, const UNARY_FUNC& f)
  {
      f(x.get_head());
      cons_for_each(x.get_tail(), f);
  }

  /**********************************************************************
  *  cons_for_each(cons<H1, T1>, cons<H2, T2>, BINARY_FUNC)		*
  **********************************************************************/
  template <class BINARY_FUNC> inline void
  cons_for_each(null_type, null_type, BINARY_FUNC)
  {
  }
  template <class H1, class T1, class H2, class T2, class BINARY_FUNC>
  inline void
  cons_for_each(cons<H1, T1>& x, const cons<H2, T2>& y, const BINARY_FUNC& f)
  {
      f(x.get_head(), y.get_head());
      cons_for_each(x.get_tail(), y.get_tail(), f);
  }
    
  /**********************************************************************
  *  make_cons(HEAD, TAIL)						*
  **********************************************************************/
  template <class HEAD, class TAIL> inline cons<HEAD, TAIL>
  make_cons(const HEAD& head, const TAIL& tail)
  {
      return cons<HEAD, TAIL>(head, tail);
  }
    
  /**********************************************************************
  *  cons_transform(cons<HEAD, TAIL>, UNARY_FUNC)			*
  **********************************************************************/
  template <class UNARY_FUNC> inline null_type
  cons_transform(null_type, UNARY_FUNC)
  {
      return null_type();
  }
  template <class HEAD, class TAIL, class UNARY_FUNC> inline auto
  cons_transform(const cons<HEAD, TAIL>& x, const UNARY_FUNC& f)
      -> decltype(make_cons(f(x.get_head()), cons_transform(x.get_tail(), f)))
  {
      return make_cons(f(x.get_head()), cons_transform(x.get_tail(), f));
  }
  template <class HEAD, class TAIL, class UNARY_FUNC> inline auto
  cons_transform(cons<HEAD, TAIL>& x, const UNARY_FUNC& f)
      -> decltype(make_cons(f(x.get_head()), cons_transform(x.get_tail(), f)))
  {
      return make_cons(f(x.get_head()), cons_transform(x.get_tail(), f));
  }

  /**********************************************************************
  *  cons_transform(cons<H1, T1>, cons<H2, T2>, BINARY_FUNC)		*
  **********************************************************************/
  template <class BINARY_FUNC> inline null_type
  cons_transform(null_type, null_type, BINARY_FUNC)
  {
      return null_type();
  }
  template <class H1, class T1, class H2, class T2, class BINARY_FUNC>
  inline auto
  cons_transform(const cons<H1, T1>& x,
		 const cons<H2, T2>& y, const BINARY_FUNC& f)
      -> decltype(make_cons(f(x.get_head(), y.get_head()),
			    cons_transform(x.get_tail(), y.get_tail(), f)))
  {
      return make_cons(f(x.get_head(), y.get_head()),
		       cons_transform(x.get_tail(), y.get_tail(), f));
  }

  /**********************************************************************
  *  cons_transform(cons<H1, T1>, cons<H2, T2>,				*
  *		    cons<H3, T3>, TRINARY_FUNC)				*
  **********************************************************************/
  template <class TRINARY_FUNC> inline null_type
  cons_transform(null_type, null_type, null_type, TRINARY_FUNC)
  {
      return null_type();
  }
  template <class H1, class T1, class H2, class T2,
	    class H3, class T3, class TRINARY_FUNC>
  inline auto
  cons_transform(const cons<H1, T1>& x, const cons<H2, T2>& y,
	    const cons<H3, T3>& z, const TRINARY_FUNC& f)
      -> decltype(make_cons(f(x.get_head(), y.get_head(), z.get_head()),
			    cons_transform(x.get_tail(),
					   y.get_tail(), z.get_tail(), f)))
  {
      return make_cons(f(x.get_head(), y.get_head(), z.get_head()),
		       cons_transform(x.get_tail(),
				      y.get_tail(), z.get_tail(), f));
  }

  /**********************************************************************
  *  Arithmetic operators						*
  **********************************************************************/
  template <class HEAD, class TAIL> inline auto
  operator -(const cons<HEAD, TAIL>& x)
      -> decltype(cons_transform(x, TU::generic_function<std::negate>()))
  {
      return cons_transform(x, TU::generic_function<std::negate>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator +(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_binary_function<TU::plus>()))
  {
      return cons_transform(x, y, TU::generic_binary_function<TU::plus>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator -(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
				 TU::generic_binary_function<TU::minus>()))
  {
      return cons_transform(x, y, TU::generic_binary_function<TU::minus>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator *(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
			    TU::generic_binary_function<TU::multiplies>()))
  {
      return cons_transform(x, y,
			    TU::generic_binary_function<TU::multiplies>());
  }

  template <class T, class HEAD, class TAIL,
	    class=typename std::enable_if<!is_tuple<T>::value>::type>
  inline auto
  operator *(const T& c, const cons<HEAD, TAIL>& x)
      -> decltype(
	  cons_transform(
	      x, std::bind(TU::generic_binary_function<TU::multiplies>(),
			   c, std::placeholders::_1)))
  {
      return cons_transform(
		 x, std::bind(TU::generic_binary_function<TU::multiplies>(),
			      c, std::placeholders::_1));
  }

  template <class HEAD, class TAIL, class T,
	    class=typename std::enable_if<!is_tuple<T>::value>::type>
  inline auto
  operator *(const cons<HEAD, TAIL>& x, const T& c)
      -> decltype(
	  cons_transform(
	      x, std::bind(TU::generic_binary_function<TU::multiplies>(),
			   std::placeholders::_1, c)))
  {
      return cons_transform(
		 x, std::bind(TU::generic_binary_function<TU::multiplies>(),
			      std::placeholders::_1, c));
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator /(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
				 TU::generic_binary_function<TU::divides>()))
  {
      return cons_transform(x, y, TU::generic_binary_function<TU::divides>());
  }
    
  template <class HEAD, class TAIL, class T,
	    class=typename std::enable_if<!is_tuple<T>::value>::type>
  inline auto
  operator /(const cons<HEAD, TAIL>& x, const T& c)
      -> decltype(cons_transform(
		      x, std::bind(TU::generic_binary_function<TU::divides>(),
				   std::placeholders::_1, c)))
  {
      return cons_transform(
		 x, std::bind(TU::generic_binary_function<TU::divides>(),
			      std::placeholders::_1, c));
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator %(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::modulus>()))
  {
      return cons_transform(x, y, TU::generic_function<std::modulus>());
  }

  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator +=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::plus_assign>());
      return y;
  }

  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator -=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::minus_assign>());
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator *=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::multiplies_assign>());
      return y;
  }
    
  template <class HEAD, class TAIL, class T>
  inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
  operator *=(cons<HEAD, TAIL>& y, const T& c)
  {
      cons_for_each(
	  y, std::bind(TU::generic_binary_function<TU::multiplies_assign>(),
		       std::placeholders::_1, c));
      return y;
  }
    
  template <class HEAD, class TAIL, class T>
  inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
  operator *=(cons<HEAD, TAIL>&& y, const T& c)
  {
      cons_for_each(
	  y, std::bind(TU::generic_binary_function<TU::multiplies_assign>(),
		       std::placeholders::_1, c));
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator /=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::divides_assign>());
      return y;
  }
    
  template <class HEAD, class TAIL, class T>
  inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
  operator /=(cons<HEAD, TAIL>& y, const T& c)
  {
      cons_for_each(
	  y, std::bind(TU::generic_binary_function<TU::divides_assign>(),
		       std::placeholders::_1, c));
      return y;
  }
    
  template <class HEAD, class TAIL, class T>
  inline typename std::enable_if<!is_tuple<T>::value, cons<HEAD, TAIL>&>::type
  operator /=(cons<HEAD, TAIL>&& y, const T& c)
  {
      cons_for_each(
	  y, std::bind(TU::generic_binary_function<TU::divides_assign>(),
		       std::placeholders::_1, c));
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator %=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::modulus_assign>());
      return y;
  }
    
  /**********************************************************************
  *  Bit operators							*
  **********************************************************************/
  template <class H1, class T1, class H2, class T2> inline auto
  operator &(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::bit_and>()))
  {
      return cons_transform(x, y, TU::generic_function<std::bit_and>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator |(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::bit_or>()))
  {
      return cons_transform(x, y, TU::generic_function<std::bit_or>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator ^(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::bit_xor>()))
  {
      return cons_transform(x, y, TU::generic_function<std::bit_xor>());
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator &=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::bit_and_assign>());
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator |=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::bit_or_assign>());
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<
      is_tuple<typename std::decay<L>::type>::value, L&>::type
  operator ^=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(y, x, TU::generic_binary_function<TU::bit_xor_assign>());
      return y;
  }
    
  /**********************************************************************
  *  Logical operators							*
  **********************************************************************/
  template <class HEAD, class TAIL> inline auto
  operator !(const cons<HEAD, TAIL>& x)
      -> decltype(cons_transform(x, TU::generic_function<std::logical_not>()))
  {
      return cons_transform(x, TU::generic_function<std::logical_not>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator &&(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
				 TU::generic_function<std::logical_and>()))
  {
      return cons_transform(x, y, TU::generic_function<std::logical_and>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator ||(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::logical_or>()))
  {
      return cons_transform(x, y, TU::generic_function<std::logical_or>());
  }
    
  /**********************************************************************
  *  Relational operators						*
  **********************************************************************/
  template <class H1, class T1, class H2, class T2> inline auto
  operator ==(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::equal_to>()))
  {
      return cons_transform(x, y, TU::generic_function<std::equal_to>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator !=(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
				 TU::generic_function<std::not_equal_to>()))
  {
      return cons_transform(x, y, TU::generic_function<std::not_equal_to>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator <(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::less>()))
  {
      return cons_transform(x, y, TU::generic_function<std::less>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator >(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::greater>()))
  {
      return cons_transform(x, y, TU::generic_function<std::greater>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator <=(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y, TU::generic_function<std::less_equal>()))
  {
      return cons_transform(x, y, TU::generic_function<std::less_equal>());
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator >=(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(x, y,
				 TU::generic_function<std::greater_equal>()))
  {
      return cons_transform(x, y, TU::generic_function<std::greater_equal>());
  }

  /**********************************************************************
  *  Selection								*
  **********************************************************************/
  template <class H1, class T1, class H2, class T2, class H3, class T3>
  inline auto
  select(const cons<H1, T1>& s, const cons<H2, T2>& x, const cons<H3, T3>& y)
      -> decltype(cons_transform(s, x, y, TU::generic_select()))
  {
      return cons_transform(s, x, y, TU::generic_select());
  }

  template <class H1, class T1, class S, class H3, class T3,
	    class=typename std::enable_if<!is_tuple<S>::value>::type>
  inline auto
  select(const cons<H1, T1>& s, const S& x, const cons<H3, T3>& y)
      -> decltype(cons_transform(s, y, std::bind(TU::generic_select(),
					    std::placeholders::_1, x,
					    std::placeholders::_2)))
  {
      return cons_transform(s, y, std::bind(TU::generic_select(),
				       std::placeholders::_1, x,
				       std::placeholders::_2));
  }

  template <class H1, class T1, class H2, class T2, class T,
	    class=typename std::enable_if<!is_tuple<T>::value>::type>
  inline auto
  select(const cons<H1, T1>& s, const cons<H2, T2>& x, const T& y)
      -> decltype(cons_transform(s, x, std::bind(TU::generic_select(),
					    std::placeholders::_1,
					    std::placeholders::_2, y)))
  {
      return cons_transform(s, x, std::bind(TU::generic_select(),
					    std::placeholders::_1,
					    std::placeholders::_2, y));
  }

}	// End of namespace boost::tuples

/************************************************************************
*  typedef htuple<T, N>							*
************************************************************************/
template <size_t N, class T> inline auto
make_contiguous_htuple(T&& x)
    -> decltype(
	tuples::make_contiguous_htuple(std::forward<T>(x),
				       TU::make_index_sequence<N>()))
{
    return tuples::make_contiguous_htuple(std::forward<T>(x),
					  TU::make_index_sequence<N>());
}

template <class T, size_t N>
using	htuple = decltype(make_contiguous_htuple<N>(std::declval<T>()));

}	// End of namespace boost

namespace TU
{
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
				TU::make_index_sequence<
				    1 + boost::tuples::length<TAIL>::value>());
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    template <class TUPLE, size_t ...IDX>
    result_type	exec(const TUPLE& arg, TU::index_sequence<IDX...>) const
		{
		    return _func(boost::get<IDX>(arg)...);
		}
    
  private:
    const FUNC&	_func;
};

template <class FUNC> inline unarizer<FUNC>
make_unarizer(const FUNC& func)
{
    return unarizer<FUNC>(func);
}

/************************************************************************
*  struct tuple_head<T>							*
************************************************************************/
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
      : tuple_head<
            typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, T)>::inherited>
  {
  };
}
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = typename detail::tuple_head<T>::type;

/************************************************************************
*  struct tuple_replace<S, T>						*
************************************************************************/
namespace detail
{
  template <class T, class S>
  struct tuple_replace : std::conditional<std::is_void<T>::value, S, T>	{};
  template <class T>
  struct tuple_replace<T, boost::tuples::null_type>
  {
      typedef boost::tuples::null_type			type;
  };
  template <class T, class HEAD, class TAIL>
  struct tuple_replace<T, boost::tuples::cons<HEAD, TAIL> >
  {
      typedef boost::tuples::cons<
	  typename tuple_replace<T, HEAD>::type,
	  typename tuple_replace<T, TAIL>::type>	type;
  };
  template <class T, BOOST_PP_ENUM_PARAMS(10, class S)>
  struct tuple_replace<T, boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)> >
      : tuple_replace<T,
            typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)>::inherited>
  {
  };
}
    
//! 与えられた型がtupleならばその全要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T=void>
using tuple_replace = typename detail::tuple_replace<T, S>::type;
    
}	// End of namespace TU
#endif	// !__TU_TUPLE_H
