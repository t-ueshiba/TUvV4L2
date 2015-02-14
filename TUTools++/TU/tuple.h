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
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <tuple>
#include <utility>			// std::forward()
#include <iostream>
#include "TU/functional.h"

namespace std
{
namespace detail
{
  /**********************************************************************
  *  struct is_tuple<T>							*
  **********************************************************************/
  template <class ...T>
  static true_type	tuple_check(tuple<T...>)		;
  static false_type	tuple_check(...)			;
    
  template <class T>
  struct is_tuple : decltype(detail::tuple_check(declval<T>()))	{};

  /**********************************************************************
  *  typedef TupleIndices<N>						*
  **********************************************************************/
  template <size_t ...> struct Indices	{};
    
  template <size_t N, size_t I=0, size_t ...IDX>
  struct MakeIndices : MakeIndices<N, I + 1, IDX..., I>
  {
  };
  template <size_t N, size_t ...IDX>
  struct MakeIndices<N, N, IDX...>
  {
      typedef Indices<IDX...>	type;
  };

  template <size_t N>
  using TupleIndices = typename MakeIndices<N>::type;
    
  /**********************************************************************
  *  make_contiguous_htuple(T, Indices<IDX...>)				*
  **********************************************************************/
  template <class T, size_t ...IDX> static inline auto
  make_contiguous_htuple(T&& x, Indices<IDX...>)
      -> decltype(make_tuple((x + IDX)...))
  {
      return make_tuple((x + IDX)...);
  }

  /**********************************************************************
  *  for_each(TUPLE, UNARY_FUNC, Indices<IDX...>)			*
  **********************************************************************/
  template <class TUPLE, class UNARY_FUNC, size_t I, size_t ...IDX> inline void
  for_each_impl(TUPLE& x, const UNARY_FUNC& f, Indices<I, IDX...>)
  {
      f(get<I>(x));
      for_each_impl(x, f, Indices<IDX...>());
  }
  template <class TUPLE, class UNARY_FUNC> inline void
  for_each_impl(TUPLE&, const UNARY_FUNC&, Indices<>)
  {
  }

  template <class TUPLE, class UNARY_FUNC,
	    class=typename enable_if<
		is_tuple<typename decay<TUPLE>::type>::value>::type>
  inline void
  for_each(TUPLE&& x, const UNARY_FUNC& f)
  {
      for_each_impl(x, f,
		    TupleIndices<
		        tuple_size<typename decay<TUPLE>::type>::value>());
  }

  /**********************************************************************
  *  for_each(TUPLE0, TUPLE1, BINARY_FUNC, Indices<IDX...>)		*
  **********************************************************************/
  template <class TUPLE0, class TUPLE1,
	    class BINARY_FUNC, size_t I, size_t ...IDX> inline void
  for_each_impl(TUPLE0& x, TUPLE1& y, const BINARY_FUNC& f, Indices<I, IDX...>)
  {
      f(get<I>(x), get<I>(y));
      for_each_impl(x, y, f, Indices<IDX...>());
  }
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC> inline void
  for_each_impl(TUPLE0&, TUPLE1&, const BINARY_FUNC&, Indices<>)
  {
  }

  template <class TUPLE0, class TUPLE1, class BINARY_FUNC,
	    class=typename enable_if<
		(is_tuple<typename decay<TUPLE0>::type>::value &&
		 is_tuple<typename decay<TUPLE1>::type>::value)>::type>
  inline void
  for_each(TUPLE0&& x, TUPLE1&& y, const BINARY_FUNC& f)
  {
      for_each_impl(x, y, f,
		    TupleIndices<
			tuple_size<typename decay<TUPLE0>::type>::value>());
  }

  /**********************************************************************
  *  transform(tuple<T...>, UNARY_FUNC)					*
  **********************************************************************/
  template <class ...T, class UNARY_FUNC, size_t ...IDX> inline auto
  transform_impl(const tuple<T...>& x, const UNARY_FUNC& f, Indices<IDX...>)
      -> decltype(make_tuple(f(get<IDX>(x))...))
  {
      return make_tuple(f(get<IDX>(x))...);
  }

  template <class ...T, class UNARY_FUNC> inline auto
  transform(const tuple<T...>& x, const UNARY_FUNC& f)
      -> decltype(transform_impl(x, f, detail::TupleIndices<sizeof...(T)>()))
  {
      return transform_impl(x, f, TupleIndices<sizeof...(T)>());
  }
    
  /**********************************************************************
  *  transform(tuple<S...>, tuple<T...>, BINARY_FUNC)			*
  **********************************************************************/
  template <class ...S, class ...T, class BINARY_FUNC, size_t ...IDX>
  inline auto
  transform_impl(const tuple<S...>& x, const tuple<T...>& y,
		 const BINARY_FUNC& f, Indices<IDX...>)
      -> decltype(make_tuple(f(get<IDX>(x), get<IDX>(y))...))
  {
      return make_tuple(f(get<IDX>(x), get<IDX>(y))...);
  }

  template <class ...S, class ...T, class BINARY_FUNC> inline auto
  transform(const tuple<S...>& x, const tuple<T...>& y, const BINARY_FUNC& f)
      -> decltype(transform_impl(x, y, f, TupleIndices<sizeof...(S)>()))
  {
      return transform_impl(x, y, f, TupleIndices<sizeof...(S)>());
  }

  /**********************************************************************
  *  transform(tuple<T...>, tuple<T...>,				*
  *	       tuple<U...>, TRINARY_FUNC)				*
  **********************************************************************/
  template <class ...S, class ...T, class ...U,
	    class TRINARY_FUNC, size_t ...IDX>
  inline auto
  transform_impl(const tuple<S...>& x, const tuple<T...>& y,
		 const tuple<U...>& z, const TRINARY_FUNC& f, Indices<IDX...>)
      -> decltype(make_tuple(f(get<IDX>(x), get<IDX>(y), get<IDX>(z))...))
  {
      return make_tuple(f(get<IDX>(x), get<IDX>(y), get<IDX>(z))...);
  }

  template <class ...S, class ...T, class ...U, class BINARY_FUNC> inline auto
  transform(const tuple<S...>& x, const tuple<T...>& y,
	    const tuple<U...>& z, const BINARY_FUNC& f)
      -> decltype(transform_impl(x, y, z, f, TupleIndices<sizeof...(S)>()))
  {  
      return transform_impl(x, y, z, f, TupleIndices<sizeof...(S)>());
  }

  /**********************************************************************
  *  struct generic_function<FUNC>					*
  **********************************************************************/
  template <template <class...> class FUNC>
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

  /**********************************************************************
  *  struct generic_binary_function<FUNC>				*
  **********************************************************************/
  template <template <class, class> class FUNC>
  struct generic_binary_function
  {
      template <class S_, class T_> auto
      operator ()(const S_& x, const T_& y) const
	  -> decltype(FUNC<S_, T_>()(x, y))
      {
	  return FUNC<S_, T_>()(x, y);
      }
  };

  /**********************************************************************
  *  struct generic_assgin<ASSIGN>					*
  **********************************************************************/
  template <template <class, class> class ASSIGN>
  struct generic_assign
  {
      template <class S_, class T_> T_&
      operator ()(const S_& x, T_&& y) const
      {
	  return ASSIGN<S_, T_>()(x, forward<T_>(y));
      }
  };

  /**********************************************************************
  *  struct generic_select						*
  **********************************************************************/
  struct generic_select
  {
      template <class S_, class T_> const T_&
      operator ()(const S_& s, const T_& x, const T_& y) const
      {
	  return select(s, x, y);
      }
  };

  /**********************************************************************
  *  struct generic_put							*
  **********************************************************************/
  struct generic_put
  {
      generic_put(ostream& out)	:_out(out)	{}
    
      template <class T_>
      void	operator ()(const T_& x) const	{ _out << ' ' << x; }

    private:
      ostream&	_out;
  };
}

/************************************************************************
*  typedef htuple<T, N>							*
************************************************************************/
template <size_t N, class T> inline auto
make_contiguous_htuple(const T& x)
    -> decltype(detail::make_contiguous_htuple(x, detail::TupleIndices<N>()))
{
    return detail::make_contiguous_htuple(x, detail::TupleIndices<N>());
}

template <class T, size_t N>
using htuple = decltype(make_contiguous_htuple<N>(declval<const T&>()));

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class ...T> inline auto
operator -(const tuple<T...>& x)
    -> decltype(detail::transform(x, detail::generic_function<negate>()))
{
    return detail::transform(x, detail::generic_function<negate>());
}
    
template <class ...S, class ...T> inline auto
operator +(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y,
				  detail::generic_binary_function<TU::plus>()))
{
    return detail::transform(x, y, detail::generic_binary_function<TU::plus>());
}

template <class ...S, class ...T> inline auto
operator -(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y,
				  detail::generic_binary_function<TU::minus>()))
{
    return detail::transform(x, y, detail::generic_binary_function<TU::minus>());
}

template <class ...S, class ...T> inline auto
operator *(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(
		    x, y, detail::generic_binary_function<TU::multiplies>()))
{
    return detail::transform(x, y,
			     detail::generic_binary_function<TU::multiplies>());
}

template <class S, class ...T,
	  class=typename enable_if<!detail::is_tuple<S>::value>::type>
inline auto
operator *(const S& c, const tuple<T...>& x)
    -> decltype(detail::transform(
		    x, bind(detail::generic_binary_function<TU::multiplies>(),
			    c, placeholders::_1)))
{
    return detail::transform(
	       x, bind(detail::generic_binary_function<TU::multiplies>(),
		       c, placeholders::_1));
}

template <class ...S, class T,
	  class=typename enable_if<!detail::is_tuple<T>::value>::type>
inline auto
operator *(const tuple<S...>& x, const T& c)
    -> decltype(detail::transform(
		    x,
		    bind(detail::generic_binary_function<TU::multiplies>(),
			      placeholders::_1, c)))
{
    return detail::transform(
	       x, bind(detail::generic_binary_function<TU::multiplies>(),
		       placeholders::_1, c));
}
    
template <class ...S, class ...T> inline auto
operator /(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(
		    x, y, detail::generic_binary_function<TU::divides>()))
{
    return detail::transform(x, y,
			     detail::generic_binary_function<TU::divides>());
}
    
template <class ...S, class T,
	  class=typename enable_if<!detail::is_tuple<T>::value>::type>
inline auto
operator /(const tuple<S...>& x, const T& c)
    -> decltype(detail::transform(
		    x, bind(detail::generic_binary_function<TU::divides>(),
			    placeholders::_1, c)))
{
    return detail::transform(
	       x, bind(detail::generic_binary_function<TU::divides>(),
		       placeholders::_1, c));
}
    
template <class ...S, class ...T> inline auto
operator %(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<modulus>()))
{
    return detail::transform(x, y, detail::generic_function<modulus>());
}

template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator +=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::plus_assign>());
    return y;
}

template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator -=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::minus_assign>());
    return y;
}

template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator *=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::multiplies_assign>());
    return y;
}
    
template <class ...S, class T>
inline typename enable_if<
    !detail::is_tuple<T>::value, tuple<S...>&>::type
operator *=(tuple<S...>& y, const T& c)
{
    detail::for_each(y, bind(detail::generic_assign<TU::multiplies_assign>(),
			     c, placeholders::_1));
    return y;
}
    
template <class ...S, class T>
inline typename enable_if<
    !detail::is_tuple<T>::value, tuple<S...>&>::type
operator *=(tuple<S...>&& y, const T& c)
{
    detail::for_each(y, bind(detail::generic_assign<TU::multiplies_assign>(),
			     c, placeholders::_1));
    return y;
}
    
template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator /=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::divides_assign>());
    return y;
}
    
template <class ...S, class T>
inline typename enable_if<
    !detail::is_tuple<T>::value, tuple<S...>&>::type
operator /=(tuple<S...>& y, const T& c)
{
    detail::for_each(y, bind(detail::generic_assign<TU::divides_assign>(),
			     c, placeholders::_1));
    return y;
}
    
template <class ...S, class T>
inline typename enable_if<
    !detail::is_tuple<T>::value, tuple<S...>&>::type
operator /=(tuple<S...>&& y, const T& c)
{
    detail::for_each(y, bind(detail::generic_assign<TU::divides_assign>(),
			     c, placeholders::_1));
    return y;
}
    
template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator %=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::modulus_assign>());
    return y;
}
    
/************************************************************************
*  Bit operators							*
************************************************************************/
template <class ...S, class ...T> inline auto
operator &(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<bit_and>()))
{
    return detail::transform(x, y, detail::generic_function<bit_and>());
}
    
template <class ...S, class ...T> inline auto
operator |(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<bit_or>()))
{
    return detail::transform(x, y, detail::generic_function<bit_or>());
}
    
template <class ...S, class ...T> inline auto
operator ^(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<bit_xor>()))
{
    return detail::transform(x, y, detail::generic_function<bit_xor>());
}

template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator &=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::bit_and_assign>());
    return y;
}
    
template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator |=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::bit_or_assign>());
    return y;
}
    
template <class L, class ...T>
inline typename enable_if<
    detail::is_tuple<typename decay<L>::type>::value, L&>::type
operator ^=(L&& y, const tuple<T...>& x)
{
    detail::for_each(x, forward<L>(y),
		     detail::generic_assign<TU::bit_xor_assign>());
    return y;
}
    
/************************************************************************
*  Logical operators							*
************************************************************************/
template <class ...T> inline auto
operator !(const tuple<T...>& x)
    -> decltype(detail::transform(x, detail::generic_function<logical_not>()))
{
    return detail::transform(x, detail::generic_function<logical_not>());
}
    
template <class ...S, class ...T> inline auto
operator &&(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<logical_and>()))
{
    return detail::transform(x, y, detail::generic_function<logical_and>());
}
    
template <class ...S, class ...T> inline auto
operator ||(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<logical_or>()))
{
    return detail::transform(x, y, detail::generic_function<logical_or>());
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class ...S, class ...T> inline auto
operator ==(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<equal_to>()))
{
    return detail::transform(x, y, detail::generic_function<equal_to>());
}
    
template <class ...S, class ...T> inline auto
operator !=(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(
		    x, y, detail::generic_function<not_equal_to>()))
{
    return detail::transform(x, y, detail::generic_function<not_equal_to>());
}
    
template <class ...S, class ...T> inline auto
operator <(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<less>()))
{
    return detail::transform(x, y, detail::generic_function<less>());
}
    
template <class ...S, class ...T> inline auto
operator >(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<greater>()))
{
    return detail::transform(x, y, detail::generic_function<greater>());
}
    
template <class ...S, class ...T> inline auto
operator <=(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(x, y, detail::generic_function<less_equal>()))
{
    return detail::transform(x, y, detail::generic_function<less_equal>());
}
    
template <class ...S, class ...T> inline auto
operator >=(const tuple<S...>& x, const tuple<T...>& y)
    -> decltype(detail::transform(
		    x, y, detail::generic_function<greater_equal>()))
{
    return detail::transform(x, y, detail::generic_function<greater_equal>());
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class T> inline const T&
select(bool s, const T& x, const T& y)
{
    return (s ? x : y);
}
    
template <class ...S, class ...T, class ...U> inline auto
select(const tuple<S...>& s, const tuple<T...>& x, const tuple<U...>& y)
    -> decltype(detail::transform(s, x, y, detail::generic_select()))
{
    return detail::transform(s, x, y, detail::generic_select());
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class ...T> inline ostream&
operator <<(ostream& out, const tuple<T...>& x)
{
    out << '(';
    detail::for_each(x, detail::generic_put(out));
    out << ')';

    return out;
}
    
}	// End of namespace std

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

    template <class ...T>
    result_type	operator ()(const std::tuple<T...>& arg) const
		{
		    return exec(arg, std::detail::TupleIndices<sizeof...(T)>());
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    template <class TUPLE, size_t ...IDX>
    result_type	exec(const TUPLE& arg, std::detail::Indices<IDX...>) const
		{
		    return _func(std::get<IDX>(arg)...);
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
  struct tuple_head : impl::identity<T>					{};
  template <class ...T>
  struct tuple_head<std::tuple<T...> >
      : std::tuple_element<0, std::tuple<T...> >			{};
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
  static typename std::conditional<std::is_void<T>::value, S, T>::type
  tuple_replace(S);
  template <class T, class ...S> static auto
  tuple_replace(std::tuple<S...>)
      -> decltype(std::make_tuple((std::declval<S>(), std::declval<T>())...));
}
    
//! 与えられた型がtupleならばその全要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T=void>
using tuple_replace = decltype(detail::tuple_replace<T>(std::declval<S>()));
    
}	// End of namespace TU
#endif	// !__TU_TUPLE_H
