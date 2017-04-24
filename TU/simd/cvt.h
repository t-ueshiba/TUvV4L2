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
template <class T, bool HI=false, bool MASK=false, class... S> inline auto
cvt(const std::tuple<S...>& t)
{
    return tuple_transform([](auto x){ return cvt<T, HI, MASK>(x); }, t);
}
    
template <class T, bool MASK=false, class... S1, class... S2> inline auto
cvt(const std::tuple<S1...>& l, const std::tuple<S2...>& r)
{
    return tuple_transform([](auto x, auto y){ return cvt<T, MASK>(x, y); },
			   l, r);
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
      using I = std::conditional_t<std::is_integral<T>::value, T, C>;
      using U = std::conditional_t<
		    (std::is_same<T, S>::value || std::is_same<C, S>::value),
		    T,				// 直接target型に変換
		    std::conditional_t<
			std::is_same<I, upper_type<signed_type<S> > >::value,
			I, upper_type<S> > >;
      using L = std::conditional_t<
		    (std::is_integral<T>::value != std::is_integral<S>::value),
		    complementary_type<S>,	// sourceのcomplementary
		    std::conditional_t<
			(std::is_same<T, S>::value ||
			 std::is_same<upper_type<signed_type<T> >, S>::value),
			T, lower_type<S> > >;
      using A = std::conditional_t<
		    std::is_same<T, complementary_type<S> >::value,
		    S, upper_type<signed_type<T> > >;
  };
  template <class T, class S>
  struct cvt_adjacent_type<T, S, true>
  {
      using U = std::conditional_t<
		    (std::is_same<T, S>::value ||
		     std::is_same<complementary_mask_type<T>, S>::value),
		    T, upper_type<S> >;
      using L = std::conditional_t<
		    (std::is_integral<T>::value != std::is_integral<S>::value),
		    complementary_mask_type<S>,
		    std::conditional_t<std::is_same<T, S>::value,
				       T, lower_type<S> > >;
      using A = std::conditional_t<
		    std::is_same<T, complementary_mask_type<S> >::value,
		    S, upper_type<T> >;
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
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/cvt.h"
#elif defined(NEON)
#  include "TU/simd/arm/cvt.h"
#endif

#endif	// !__TU_SIMD_CVT_H
