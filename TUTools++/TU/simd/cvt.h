/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVT_H)
#define	__TU_SIMD_CVT_H

#include "TU/tuple.h"
#include "TU/simd/zero.h"
#include "TU/simd/cast.h"
#include "TU/simd/shift.h"
#include "TU/simd/bit_shift.h"
#include <boost/tuple/tuple_io.hpp>

namespace TU
{
namespace simd
{
/************************************************************************
*  Converting vecs							*
************************************************************************/
template <class T, bool MASK=false, class S>
inline typename std::enable_if<(vec<T>::size == vec<S>::size), vec<T> >::type
cvt(vec<S> x)
{
    return x;
}
    
//! S型ベクトルの上位または下位半分を要素数が半分のT型ベクトルに型変換する．
/*!
  S, Tは符号付き／符号なしのいずれでも良いが，符号付き -> 符号なしの変換はできない．
  \param HI	falseならば下位，trueならば上位を変換
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class T, bool MASK, bool HI, class S> vec<T>
cvt(vec<S> x)								;
	
//! 2つのS型ベクトルを要素数が2倍のT型ベクトルに型変換する．
/*!
  Sが符号付き／符号なしのいずれの場合も飽和処理が行われる．
  \param x	変換されるベクトル
  \param y	変換されるベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したベクトル
*/
template <class T, bool MASK=false, class S> vec<T>
cvt(vec<S> x, vec<S> y)							;
    
/************************************************************************
*  Converting vec tuples						*
************************************************************************/
namespace detail
{
  template <class T, bool MASK, bool HI>
  struct generic_cvt
  {
      template <class S_>
      typename std::enable_if<(vec<T>::size == vec<S_>::size), vec<T> >::type
		operator ()(vec<S_> x) const
		{
		    return cvt<T, MASK>(x);
		}
      template <class S_>
      typename std::enable_if<(vec<T>::size < vec<S_>::size), vec<T> >::type
		operator ()(vec<S_> x) const
		{
		    return cvt<T, MASK, HI>(x);
		}
      template <class S_>
      vec<T>	operator ()(vec<S_> x, vec<S_> y) const
		{
		    return cvt<T, MASK>(x, y);
		}
  };
}

template <class T, bool MASK=false, bool HI=false, class HEAD, class TAIL>
inline auto
cvt(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    x, detail::generic_cvt<T, MASK, HI>()))
{
    return boost::tuples::cons_transform(x, detail::generic_cvt<T, MASK, HI>());
}
    
template <class T, bool MASK=false, class H1, class T1, class H2, class T2>
inline auto
cvt(const boost::tuples::cons<H1, T1>& x, const boost::tuples::cons<H2, T2>& y)
    -> decltype(boost::tuples::cons_transform(
		    x, y, detail::generic_cvt<T, MASK, 0>()))
{
    return boost::tuples::cons_transform(
	       x, y, detail::generic_cvt<T, MASK, 0>());
}
    
/************************************************************************
*  Adjacent target types of conversions					*
************************************************************************/
namespace detail
{
  template <class T, class S, bool MASK>
  struct cvt_adjacent_type
  {
      using C = complementary_type<T>;
      using I = typename std::conditional<std::is_integral<T>::value,
					  T, C>::type;
      using U = typename std::conditional<
		    (std::is_same<T, S>::value || std::is_same<C, S>::value),
		    T,				// 直接target型に変換
		    typename std::conditional<
			std::is_same<I, upper_type<signed_type<S> > >::value,
			upper_type<signed_type<S> >,
			upper_type<S> >::type>::type;
      using L = typename std::conditional<
		    (std::is_integral<T>::value != std::is_integral<S>::value),
		    complementary_type<S>,
		    typename std::conditional<
			(std::is_same<T, S>::value ||
			 std::is_same<upper_type<signed_type<T> >, S>::value),
			T, lower_type<S> >::type>::type;
      using A = typename std::conditional<
		    std::is_same<T, complementary_type<S> >::value,
		    S, upper_type<signed_type<T> > >::type;
  };
  template <class T, class S>
  struct cvt_adjacent_type<T, S, true>
  {
      using U = typename std::conditional<
		    (std::is_same<T, S>::value ||
		     std::is_same<complementary_mask_type<T>, S>::value),
		    T, upper_type<S> >::type;
      using L = typename std::conditional<
		    (std::is_integral<T>::value != std::is_integral<S>::value),
		    complementary_mask_type<S>,
		    typename std::conditional<std::is_same<T, S>::value,
					      T, lower_type<S> >::type>::type;
      using A = typename std::conditional<
		    std::is_same<T, complementary_mask_type<S> >::value,
		    S, upper_type<T> >::type;
  };
}
    
//! 要素数がより少ないベクトルへの多段変換において最初の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の最初の変換先の要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	最初の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_upper_type = typename detail::cvt_adjacent_type<T, S, MASK>::U;

//! 要素数がより多いベクトルへの多段変換において最初の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の最初の変換先の要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	最初の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_lower_type = typename detail::cvt_adjacent_type<T, S, MASK>::L;

//! 要素数がより多いベクトルへの多段変換において最終的な変換先の直前の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<T> に達する直前のベクトルの要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	最終的な変換先に達する直前のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_above_type = typename detail::cvt_adjacent_type<T, S, MASK>::A;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/cvt.h"
#elif defined(NEON)
#  include "TU/simd/arm/cvt.h"
#endif

#endif	// !__TU_SIMD_CVT_H
