/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVT_MASK_H)
#define	__TU_SIMD_CVT_MASK_H

#include "TU/tuple.h"
#include "TU/simd/cast.h"
#include <boost/tuple/tuple_io.hpp>

namespace TU
{
namespace simd
{
/************************************************************************
*  Mask conversion operators						*
************************************************************************/
//! S型マスクベクトルを要素数が同一のT型マスクベクトルに型変換する．
/*!
  \param x	変換されるマスクベクトル
  \return	変換されたマスクベクトル
*/
template <class T, class S>
inline typename std::enable_if<(vec<T>::size == vec<S>::size), vec<T> >::type
cvt_mask(vec<S> x)
{
    return x;	// S == T の場合の実装
}

//! S型マスクベクトルの上位または下位半分を要素数が半分のT型マスクベクトルに型変換する．
/*!
  S, Tは符号付き／符号なしのいずれでも良い．
  \param I	I=0ならば下位，I=1ならば上位を変換
  \param x	変換されるマスクベクトル
  \return	変換されたマスクベクトル
*/
template <class T, size_t I, class S> vec<T>	cvt_mask(vec<S> x)	;
	
//! 2つのS型マスクベクトルを要素数が2倍のT型マスクベクトルに型変換する．
/*!
  S, Tは符号付き／符号なしのいずれでも良い．
  \param x	変換されるマスクベクトル
  \param y	変換されるマスクベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したマスクベクトル
*/
template <class T, class S> vec<T>	cvt_mask(vec<S> x, vec<S> y)	;

/************************************************************************
*  Converting mask vec tuples						*
************************************************************************/
namespace detail
{
  template <class T, size_t I>
  struct generic_cvt_mask
  {
      template <class S_>
      typename std::enable_if<(vec<T>::size == vec<S_>::size), vec<T> >::type
		operator ()(vec<S_> x) const
		{
		    return cvt_mask<T>(x);
		}
      template <class S_>
      typename std::enable_if<(vec<T>::size < vec<S_>::size), vec<T> >::type
		operator ()(vec<S_> x) const
		{
		    return cvt_mask<T, I>(x);
		}
      template <class S_>
      vec<T>	operator ()(vec<S_> x, vec<S_> y) const
		{
		    return cvt_mask<T>(x, y);
		}
  };
}

template <class T, size_t I=0, class HEAD, class TAIL> inline auto
cvt_mask(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(x,
					      detail::generic_cvt_mask<T, I>()))
{
    return boost::tuples::cons_transform(x, detail::generic_cvt<T, I>());
}
    
template <class T, class H1, class T1, class H2, class T2> inline auto
cvt_mask(const boost::tuples::cons<H1, T1>& x,
	 const boost::tuples::cons<H2, T2>& y)
    -> decltype(boost::tuples::cons_transform(x, y,
					      detail::generic_cvt_mask<T, 0>()))
{
    return boost::tuples::cons_transform(x, y,
					 detail::generic_cvt_mask<T, 0>());
}
    
/************************************************************************
*  Adjacent target types of conversions					*
************************************************************************/
//! 要素数がより少ないマスクベクトルへの多段変換において最初の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の最初の変換先の要素型を返す.
  \param S	変換されるマスクベクトルの要素型
  \param T	最終的な変換先のマスクベクトルの要素型
  \return	最初の変換先のマスクベクトルの要素型
*/
template <class T, class S>
using cvt_mask_upper_type = typename std::conditional<
				(std::is_same<T, S>::value ||
				 std::is_same<
				     complementary_mask_type<T>, S>::value),
				T, upper_type<S> >::type;

//! 要素数がより多いマスクベクトルへの多段変換において最初の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の最初の変換先の要素型を返す.
  \param S	変換されるマスクベクトルの要素型
  \param T	最終的な変換先のマスクベクトルの要素型
  \return	最初の変換先のマスクベクトルの要素型
*/
template <class T, class S>
using cvt_mask_lower_type = typename std::conditional<
				(std::is_integral<T>::value !=
				 std::is_integral<S>::value),
				complementary_mask_type<S>,
				typename std::conditional<
				    std::is_same<T, S>::value,
				    T, lower_type<S> >::type>::type;

//! 要素数がより多いマスクベクトルへの多段変換において最終的な変換先の直前の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<T> に達する直前のマスクベクトルの要素型を返す.
  \param S	変換されるマスクベクトルの要素型
  \param T	最終的な変換先のマスクベクトルの要素型
  \return	最終的な変換先に達する直前のマスクベクトルの要素型
*/
template <class T, class S>
using cvt_mask_above_type = typename std::conditional<
				std::is_same<
				    T, complementary_mask_type<S> >::value,
				S, upper_type<T> >::type;
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/cvt_mask.h"
#elif defined(NEON)
#  include "TU/simd/arm/cvt_mask.h"
#endif

#endif	// !__TU_SIMD_CVT_MASK_H
