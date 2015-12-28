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
  *  make_uniform_htuple(T, TU::index_sequence<IDX...>)			*
  **********************************************************************/
  template <class T, size_t... IDX> static inline auto
  make_uniform_htuple(T&& x, TU::index_sequence<IDX...>)
      -> decltype(boost::make_tuple((IDX, x)...))
  {
      return boost::make_tuple((IDX, x)...);
  }

  /**********************************************************************
  *  make_contiguous_htuple(T, TU::index_sequence<IDX...>)		*
  **********************************************************************/
  template <class T, size_t... IDX> static inline auto
  make_contiguous_htuple(T&& x, TU::index_sequence<IDX...>)
      -> decltype(boost::make_tuple((x + IDX)...))
  {
      return boost::make_tuple((x + IDX)...);
  }

  /**********************************************************************
  *  make_cons(HEAD&&, TAIL&&)						*
  **********************************************************************/
  template <class HEAD, class TAIL> inline cons<HEAD, TAIL>
  make_cons(HEAD&& head, TAIL&& tail)
  {
      return cons<HEAD, TAIL>(head, tail);
  }
    
  /**********************************************************************
  *  tuple_cat(const cons<H1, T1>&, const cons<H2, T2>&)		*
  **********************************************************************/
  template <class HEAD, class TAIL> inline const cons<HEAD, TAIL>&
  tuple_cat(null_type, const cons<HEAD, TAIL>& tail)
  {
      return tail;
  }
  template <class H1, class T1, class H2, class T2> inline auto
  tuple_cat(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(make_cons(x.get_head(), tuple_cat(x.get_tail(), y)))
  {
      return make_cons(x.get_head(), tuple_cat(x.get_tail(), y));
  }

  /**********************************************************************
  *  get_head(TUPLE&&)							*
  **********************************************************************/
  template <class T>
  inline typename std::enable_if<!is_tuple<T>::value, T>::type
  get_head(T&& x)
  {
      return x;
  }
  template <class T,
	    typename std::enable_if<is_tuple<T>::value>::type* = nullptr>
  inline auto
  get_head(T&& x) -> decltype(x.get_head())
  {
      return x.get_head();
  }
    
  /**********************************************************************
  *  get_tail(TUPLE&&)							*
  **********************************************************************/
  template <class T>
  inline typename std::enable_if<!is_tuple<T>::value, T>::type
  get_tail(T&& x)
  {
      return x;
  }
  template <class T,
	    typename std::enable_if<is_tuple<T>::value>::type* = nullptr>
  inline auto
  get_tail(T&& x) -> decltype(x.get_tail())
  {
      return x.get_tail();
  }
    
  /**********************************************************************
  *  struct contains_tuple<ARGS...>					*
  **********************************************************************/
  template <class... ARGS>
  struct contains_tuple : std::false_type
  {
  };
  template <class ARG, class... ARGS>
  struct contains_tuple<ARG, ARGS...>
      : std::integral_constant<bool, (is_tuple<ARG>::value ||
				      contains_tuple<ARGS...>::value)>
  {
  };
    
  /**********************************************************************
  *  cons_for_each(FUNC, TUPLE&&)					*
  **********************************************************************/
  template <class FUNC, class... TUPLE>
  inline typename std::enable_if<!contains_tuple<TUPLE...>::value,
				 null_type>::type
  cons_for_each(FUNC, TUPLE&&...)
  {
  }
  template <class FUNC, class ...TUPLE>
  inline typename std::enable_if<contains_tuple<TUPLE...>::value>::type
  cons_for_each(FUNC f, TUPLE&&... x)
  {
      f(get_head(x)...);
      cons_for_each(f, get_tail(x)...);
  }

  /**********************************************************************
  *  cons_transform(FUNC, TUPLE&&...)					*
  **********************************************************************/
  template <class FUNC, class... TUPLE>
  inline typename std::enable_if<!contains_tuple<TUPLE...>::value,
				 null_type>::type
  cons_transform(FUNC, TUPLE&&...)
  {
      return null_type();
  }

  template <class FUNC, class... TUPLE,
	    typename std::enable_if<contains_tuple<TUPLE...>::value>::type*
	    = nullptr> inline auto
  cons_transform(FUNC f, TUPLE&&... x)
      -> decltype(make_cons(f(get_head(x)...),
			    cons_transform(f, get_tail(x)...)))
  {
      return make_cons(f(get_head(x)...), cons_transform(f, get_tail(x)...));
  }

  /**********************************************************************
  *  Arithmetic operators						*
  **********************************************************************/
  template <class HEAD, class TAIL> inline auto
  operator -(const cons<HEAD, TAIL>& x)
      -> decltype(cons_transform(TU::generic_function<std::negate>(), x))
  {
      return cons_transform(TU::generic_function<std::negate>(), x);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator +(const S& x, const T& y)
      -> decltype(cons_transform(TU::plus(), x, y))
  {
      return cons_transform(TU::plus(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator -(const S& x, const T& y)
      -> decltype(cons_transform(TU::minus(), x, y))
  {
      return cons_transform(TU::minus(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator *(const S& x, const T& y)
      -> decltype(cons_transform(TU::multiplies(), x, y))
  {
      return cons_transform(TU::multiplies(), x, y);
  }

  template <class HEAD, class TAIL, class T> inline auto
  operator /(const cons<HEAD, TAIL>& x, const T& y)
      -> decltype(cons_transform(TU::divides(), x, y))
  {
      return cons_transform(TU::divides(), x, y);
  }
    
  template <class H1, class T1, class H2, class T2> inline auto
  operator %(const cons<H1, T1>& x, const cons<H2, T2>& y)
      -> decltype(cons_transform(TU::generic_function<std::modulus>(), x, y))
  {
      return cons_transform(TU::generic_function<std::modulus>(), x, y);
  }

  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator +=(L&& y, const T& x)
  {
      cons_for_each(TU::plus_assign(), y, x);
      return y;
  }

  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator -=(L&& y, const T& x)
  {
      cons_for_each(TU::minus_assign(), y, x);
      return y;
  }
    
  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator *=(L&& y, const T& x)
  {
      cons_for_each(TU::multiplies_assign(), y, x);
      return y;
  }
    
  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator /=(L&& y, const T& x)
  {
      cons_for_each(TU::divides_assign(), y, x);
      return y;
  }
    
  template <class L, class HEAD, class TAIL>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator %=(L&& y, const cons<HEAD, TAIL>& x)
  {
      cons_for_each(TU::modulus_assign(), y, x);
      return y;
  }

  /**********************************************************************
  *  Bit operators							*
  **********************************************************************/

  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator &(const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_function<std::bit_and>(), x, y))
  {
      return cons_transform(TU::generic_function<std::bit_and>(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator |(const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_function<std::bit_or>(), x, y))
  {
      return cons_transform(TU::generic_function<std::bit_or>(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator ^(const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_function<std::bit_xor>(), x, y))
  {
      return cons_transform(TU::generic_function<std::bit_xor>(), x, y);
  }
    
  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator &=(L&& y, const T& x)
  {
      cons_for_each(TU::bit_and_assign(), y, x);
      return y;
  }
    
  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator |=(L&& y, const T& x)
  {
      cons_for_each(TU::bit_or_assign(), y, x);
      return y;
  }
    
  template <class L, class T>
  inline typename std::enable_if<is_tuple<L>::value, L&>::type
  operator ^=(L&& y, const T& x)
  {
      cons_for_each(TU::bit_xor_assign(), y, x);
      return y;
  }

  /**********************************************************************
  *  Logical operators							*
  **********************************************************************/
  template <class HEAD, class TAIL> inline auto
  operator !(const cons<HEAD, TAIL>& x)
      -> decltype(cons_transform(TU::generic_function<std::logical_not>(), x))
  {
      return cons_transform(TU::generic_function<std::logical_not>(), x);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator &&(const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_function<std::logical_and>(),
				 x, y))
  {
      return cons_transform(TU::generic_function<std::logical_and>(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator ||(const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_function<std::logical_or>(),
				 x, y))
  {
      return cons_transform(TU::generic_function<std::logical_or>(), x, y);
  }

  /**********************************************************************
  *  Relational operators						*
  **********************************************************************/
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator ==(const S& x, const T& y)
      -> decltype(cons_transform(TU::equal_to(), x, y))
  {
      return cons_transform(TU::equal_to(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator !=(const S& x, const T& y)
      -> decltype(cons_transform(TU::not_equal_to(), x, y))
  {
      return cons_transform(TU::not_equal_to(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator <(const S& x, const T& y)
      -> decltype(cons_transform(TU::less(), x, y))
  {
      return cons_transform(TU::less(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator >(const S& x, const T& y)
      -> decltype(cons_transform(TU::greater(), x, y))
  {
      return cons_transform(TU::greater(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator <=(const S& x, const T& y)
      -> decltype(cons_transform(TU::less_equal(), x, y))
  {
      return cons_transform(TU::less_equal(), x, y);
  }
    
  template <class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  operator >=(const S& x, const T& y)
      -> decltype(cons_transform(TU::greater_equal(), x, y))
  {
      return cons_transform(TU::greater_equal(), x, y);
  }

  /**********************************************************************
  *  Selection								*
  **********************************************************************/

  template <class HEAD, class TAIL, class S, class T,
	    typename std::enable_if<
		(is_tuple<S>::value || is_tuple<T>::value)>::type* = nullptr>
  inline auto
  select(const cons<HEAD, TAIL>& s, const S& x, const T& y)
      -> decltype(cons_transform(TU::generic_select(), s, x, y))
  {
      return cons_transform(TU::generic_select(), s, x, y);
  }
}	// End of namespace boost::tuples

/************************************************************************
*  typedef htuple<T, N>							*
************************************************************************/
template <size_t N, class T> inline auto
make_uniform_htuple(T&& x)
    -> decltype(tuples::make_uniform_htuple(std::forward<T>(x),
					    TU::make_index_sequence<N>()))
{
    return tuples::make_uniform_htuple(std::forward<T>(x),
				       TU::make_index_sequence<N>());
}

template <size_t N, class T> inline auto
make_contiguous_htuple(T&& x)
    -> decltype(tuples::make_contiguous_htuple(std::forward<T>(x),
					       TU::make_index_sequence<N>()))
{
    return tuples::make_contiguous_htuple(std::forward<T>(x),
					  TU::make_index_sequence<N>());
}

template <class T, size_t N>
using	htuple = decltype(boost::make_uniform_htuple<N>(std::declval<T>()));

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
    template <class TUPLE, size_t... IDX>
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
*  struct tuple_head<T>, tuple_leftmost<T>, tuple_nelms<T>		*
************************************************************************/
namespace detail
{
  template <class T>
  struct tuple_traits
  {
      static constexpr size_t	nelms = 1;
      typedef T			head_type;
      typedef T			leftmost_type;
  };
  template <>
  struct tuple_traits<boost::tuples::null_type>
  {
      static constexpr size_t	nelms = 0;
      typedef void		head_type;
      typedef void		leftmost_type;
  };
  template <class HEAD, class TAIL>
  struct tuple_traits<boost::tuples::cons<HEAD, TAIL> >
  {
      static constexpr size_t	nelms = tuple_traits<HEAD>::nelms
				      + tuple_traits<TAIL>::nelms;
      typedef HEAD						head_type;
      typedef typename tuple_traits<HEAD>::leftmost_type	leftmost_type;
  };
  template <class ...T>
  struct tuple_traits<boost::tuple<T...> >
      : tuple_traits<typename boost::tuple<T...>::inherited>
  {
  };
}
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = typename detail::tuple_traits<T>::head_type;

//! 与えられた型がtupleならばその最左要素の型を，そうでなければ元の型を返す．
/*!
  \param T	その最左要素の型を調べるべき型
*/
template <class T>
using tuple_leftmost = typename detail::tuple_traits<T>::leftmost_type;

//! 与えられた型がtupleまたはnull_typeならばその要素数を，そうでなければ1を返す．
/*!
  \param T	その最左要素の型を調べるべき型
*/
template <class T>
struct tuple_nelms
{
    static constexpr size_t	value = detail::tuple_traits<T>::nelms;
};
    
/************************************************************************
*  struct tuple_for_all<T, COND, ARGS...>				*
************************************************************************/
template <class T, template <class ...> class COND, class ...ARGS>
struct tuple_for_all : std::integral_constant<bool, COND<T, ARGS...>::value>
{
};
template <template <class ...> class COND, class ...ARGS>
struct tuple_for_all<boost::tuples::null_type, COND, ARGS...>
    : std::true_type
{
};
template <class HEAD, class TAIL,
	  template <class ...> class COND, class ...ARGS>
struct tuple_for_all<boost::tuples::cons<HEAD, TAIL>, COND,  ARGS...>
    : std::integral_constant<
	  bool,
	  (COND<HEAD, ARGS...>::value &&
	   tuple_for_all<TAIL, COND, ARGS...>::value)>
{
};
template <BOOST_PP_ENUM_PARAMS(10, class S),
	  template <class ...> class COND, class ...ARGS>
struct tuple_for_all<boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)>,
		     COND, ARGS...>
    : tuple_for_all<
	  typename boost::tuple<BOOST_PP_ENUM_PARAMS(10, S)>::inherited,
	  COND, ARGS...>
{
};
    
/************************************************************************
*  struct tuple_is_uniform<T>						*
************************************************************************/
template <class T>
struct tuple_is_uniform : std::true_type
{
};
template <class HEAD, class TAIL>
struct tuple_is_uniform<boost::tuples::cons<HEAD, TAIL> >
    : std::integral_constant<
	  bool,
	  (std::is_same<TAIL, boost::tuples::null_type>::value ||
	   (std::is_same<HEAD, tuple_head<TAIL> >::value &&
	    tuple_is_uniform<TAIL>::value))>
{
};
template <class ...S>
struct tuple_is_uniform<boost::tuple<S...> >
    : tuple_is_uniform<typename boost::tuple<S...>::inherited>
{
};
    
/************************************************************************
*  struct tuple_replace<S, T>						*
************************************************************************/
namespace detail
{
  template <class T, class S>
  struct tuple_replace : std::conditional<std::is_void<T>::value, S, T>
  {
  };
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
  template <class T, class ...S>
  struct tuple_replace<T, boost::tuple<S...> >
      : tuple_replace<T, typename boost::tuple<S...>::inherited>
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
