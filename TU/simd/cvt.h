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
//! S型ベクトルの上位または下位半分を要素数が半分のT型ベクトルに型変換する．
/*!
  S, Tは符号付き／符号なしのいずれでも良いが，符号付き -> 符号なしの変換はできない．
  \param HI	falseならば下位，trueならば上位を変換
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class T, bool HI=false, bool MASK=false, class S>
vec<T>	cvt(vec<S> x)							;
	
//! 2つのS型ベクトルを要素数が2倍のT型ベクトルに型変換する．
/*!
  Sが符号付き／符号なしのいずれの場合も飽和処理が行われる．
  \param x	変換されるベクトル
  \param y	変換されるベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したベクトル
*/
template <class T, bool MASK=false, class S>
vec<T>	cvt(vec<S> x, vec<S> y)						;
    
/************************************************************************
*  Converting vec tuples						*
************************************************************************/
namespace detail
{
  template <class T, bool HI, bool MASK>
  struct generic_cvt
  {
      template <class S_>
      vec<T>	operator ()(vec<S_> x) const
		{
		    return cvt<T, HI, MASK>(x);
		}
      template <class S_>
      vec<T>	operator ()(vec<S_> x, vec<S_> y) const
		{
		    return cvt<T, MASK>(x, y);
		}
  };
}

template <class T, bool HI=false, bool MASK=false, class HEAD, class TAIL>
inline auto
cvt(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    detail::generic_cvt<T, HI, MASK>(), x))
{
    return boost::tuples::cons_transform(detail::generic_cvt<T, HI, MASK>(), x);
}
    
template <class T, bool MASK=false, class H1, class T1, class H2, class T2>
inline auto
cvt(const boost::tuples::cons<H1, T1>& x, const boost::tuples::cons<H2, T2>& y)
    -> decltype(boost::tuples::cons_transform(
		    detail::generic_cvt<T, false, MASK>(), x, y))
{
    return boost::tuples::cons_transform(
	       detail::generic_cvt<T, false, MASK>(), x, y);
}
    
/************************************************************************
*  Adjacent target types of conversions					*
************************************************************************/
namespace detail
{
  template <class T, class S, bool MASK>
  struct cvt_adjacent_type
  {
      using C = complementary_type<T>;		// targetのcomplementary
      using I = typename std::conditional<std::is_integral<T>::value,
					  T, C>::type;
      using U = typename std::conditional<
		    (std::is_same<T, S>::value || std::is_same<C, S>::value),
		    T,				// 直接target型に変換
		    typename std::conditional<
			std::is_same<I, upper_type<signed_type<S> > >::value,
			I, upper_type<S> >::type>::type;
      using L = typename std::conditional<
		    (std::is_integral<T>::value != std::is_integral<S>::value),
		    complementary_type<S>,	// sourceのcomplementary
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
    
//! 要素数がより少ないベクトルへの多段変換において直後の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の直後の変換先の要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	直後の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_upper_type = typename detail::cvt_adjacent_type<T, S, MASK>::U;

//! 要素数がより多いベクトルへの多段変換において直後の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の直後の変換先の要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	直後の変換先のベクトルの要素型
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

//! 異なる型の要素を持つベクトルへの多段変換において直後の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の直後の変換先の要素型を返す.
  \param S	変換されるベクトルの要素型
  \param T	最終的な変換先のベクトルの要素型
  \return	直後の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_adjacent_type = typename std::conditional<
			      (vec<T>::size < vec<S>::size),
			      cvt_upper_type<T, S, MASK>,
			      typename std::conditional<
				  (vec<cvt_lower_type<T, S, MASK> >::size >
				   vec<S>::size),
			      S, cvt_lower_type<T, S, MASK> >::type>::type;

/************************************************************************
*  Converting vecs to adjacent type					*
************************************************************************/
template <class T, bool HI=false, bool MASK=false, class S>
inline vec<cvt_adjacent_type<T, S, MASK> >
cvtadj(vec<S> x)
{
    return cvt<cvt_adjacent_type<T, S, MASK>, HI, MASK>(x);
}
    
template <class T, bool MASK=false, class S>
inline vec<cvt_lower_type<T, S, MASK> >
cvtadj(vec<S> x, vec<S> y)
{
    return cvt<cvt_lower_type<T, S, MASK>, MASK>(x, y);
}
    
/************************************************************************
*  Converting vec tuples to adjacent type				*
************************************************************************/
namespace detail
{
  template <class T, bool HI, bool MASK>
  struct generic_cvtadj
  {
      template <class S_>
      auto	operator ()(vec<S_> x) const
		    -> decltype(cvtadj<T, HI, MASK>(x))
		{
		    return cvtadj<T, HI, MASK>(x);
		}
      template <class S_>
      auto	operator ()(vec<S_> x, vec<S_> y) const
		    -> decltype(cvtadj<T, MASK>(x, y))
		{
		    return cvtadj<T, MASK>(x, y);
		}
  };
}

template <class T, bool HI=false, bool MASK=false, class HEAD, class TAIL>
inline auto
cvtadj(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    detail::generic_cvtadj<T, HI, MASK>(), x))
{
    return boost::tuples::cons_transform(
	       detail::generic_cvtadj<T, HI, MASK>(), x);
}
    
template <class T, bool MASK=false, class H1, class T1, class H2, class T2>
inline auto
cvtadj(const boost::tuples::cons<H1, T1>& x,
       const boost::tuples::cons<H2, T2>& y)
    -> decltype(boost::tuples::cons_transform(
		    detail::generic_cvtadj<T, false, MASK>(), x, y))
{
    return boost::tuples::cons_transform(
	       detail::generic_cvtadj<T, false, MASK>(), x, y);
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/cvt.h"
#elif defined(NEON)
#  include "TU/simd/arm/cvt.h"
#endif

#endif	// !__TU_SIMD_CVT_H
