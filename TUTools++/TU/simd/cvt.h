/*!
  \file		cvt.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトル間の型変換関数の定義
*/
#if !defined(TU_SIMD_CVT_H)
#define	TU_SIMD_CVT_H

#include "TU/iterator.h"
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
//! S型ベクトルの上位または下位半分を1つ上位(要素数が半分)のT型ベクトルに型変換する．
/*!
  S, Tは符号付き／符号なしのいずれでも良いが，符号付き -> 符号なしの変換はできない．
  \param T	変換先のベクトルの要素型
  \param HI	falseならば下位，trueならば上位を変換
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class T, bool HI=false, bool MASK=false, class S>
vec<T>	cvt(vec<S> x)							;
	
//! 2つのS型ベクトルを1つ下位(要素数が2倍)のT型ベクトルに型変換する．
/*!
  Sが符号付き／符号なしのいずれの場合も飽和処理が行われる．
  \param T	変換先のベクトルの要素型
  \param MASK	falseならば数値ベクトルとして，
		trueならばマスクベクトルとして変換，
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
template <class T, bool HI=false, bool MASK=false, class... VEC> inline auto
cvt(const std::tuple<VEC...>& t)
{
    return tuple_transform([](auto x){ return cvt<T, HI, MASK>(x); }, t);
}

template <class T, bool MASK=false, class... VEC> inline auto
cvt(const std::tuple<VEC...>& l, const std::tuple<VEC...>& r)
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
    
//! より上位の要素を持つベクトルへの多段変換において直後の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の直後の変換先の要素型を返す.
  \param T	最終的な変換先のベクトルの要素型
  \param S	変換されるベクトルの要素型
  \param MASK	falseならば数値ベクトルとして，
		trueならばマスクベクトルとして変換，
  \param MASK	trueならばマスクベクトルとして変換，
		falseならば数値ベクトルとして変換
  \return	直後の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_upper_type = typename detail::cvt_adjacent_type<T, S, MASK>::U;

//! より下位の要素を持つベクトルへの多段変換において直後の変換先の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<S> の直後の変換先の要素型を返す.
  \param T	最終的な変換先のベクトルの要素型
  \param S	変換されるベクトルの要素型
  \param MASK	falseならば数値ベクトルとして，
		trueならばマスクベクトルとして変換，
  \return	直後の変換先のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_lower_type = typename detail::cvt_adjacent_type<T, S, MASK>::L;

//! より下位の要素を持つベクトルへの多段変換において最終的な変換先の直上の要素型を返す.
/*!
  vec<S> を vec<T> に変換する過程で vec<T> に達する直前のベクトルの要素型を返す.
  \param T	最終的な変換先のベクトルの要素型
  \param S	変換されるベクトルの要素型
  \param MASK	falseならば数値ベクトルとして，
		trueならばマスクベクトルとして変換，
  \return	最終的な変換先の直上のベクトルの要素型
*/
template <class T, class S, bool MASK>
using cvt_above_type = typename detail::cvt_adjacent_type<T, S, MASK>::A;
    
/************************************************************************
*  Converting vecs or vec tuples to upper adjacent types		*
************************************************************************/
//! S型ベクトルの上位または下位半分を直上位または同位の隣接ベクトルに型変換する．
/*!
  S型ベクトルをT型ベクトルへ多段変換する過程の1ステップとして，
  S型ベクトルを直上位または同位の隣接ベクトルに変換する．
  \param T	最終的な変換先のベクトルの要素型
  \param HI	falseならば下位，trueならば上位を変換
  \param MASK	falseならば数値ベクトルとして，
		trueならばマスクベクトルとして変換，
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class T, bool HI=false, bool MASK=false, size_t=0, class S>
inline auto
cvtup(vec<S> x)
{
    return cvt<cvt_upper_type<T, S, MASK>, HI, MASK>(x);
}

template <class T, bool HI, bool MASK, size_t N, class ITER,
	  std::enable_if_t<iterator_value<ITER>::size == N>* = nullptr>
inline auto
cvtup(ITER& iter)
{
    return *iter++;
}

template <class T, bool HI, bool MASK, size_t N, class ITER>
inline std::enable_if_t<iterator_value<ITER>::size != N, ITER&>
cvtup(ITER& iter)
{
    return iter;
}

template <class T, bool HI=false, bool MASK=false, size_t N=0, class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
cvtup(TUPLE&& t)
{
    return tuple_transform([](auto&& x) -> decltype(auto)
			   { return cvtup<T, HI, MASK, N>(
					std::forward<decltype(x)>(x)); },
			   t);
}

/************************************************************************
*  Converting vecs or vec tuples to lower adjacent types		*
************************************************************************/
//! S型ベクトルを要素数が等しい直下位ベクトルに変換する．
/*!
  S型ベクトルをT型ベクトルへ多段変換する過程の1ステップとして，
  vec<S> とその直下位ベクトル vec<L> の要素数が等しければ
  vec<L> に変換する．そうでなければ変換せずに vec<S> のまま返す．
  \param T	最終的な変換先のベクトルの要素型
  \param MASK	trueならばマスクベクトルとして変換，
		falseならば数値ベクトルとして変換
  \param x	変換されるベクトル
  \return	変換されたベクトル
*/
template <class T, bool MASK=false, size_t=vec<T>::size, class S> inline auto
cvtdown(vec<S> x)
{
    using L = cvt_lower_type<T, S, MASK>;	// Sの直下位の要素型
    using A = std::conditional_t<vec<L>::size == vec<S>::size, L, S>;

    return cvt<A, false, MASK>(x);
}

//! 2つのS型ベクトルを要素数が2倍の直下位ベクトルに変換する．
/*!
  S型ベクトルをT型ベクトルへ多段変換する過程の1ステップとして，
  2つのS型ベクトルを1つの直下位の隣接ベクトルに変換する．
  \param T	最終的な変換先のベクトルの要素型
  \param MASK	trueならばマスクベクトルとして変換，
		falseならば数値ベクトルとして変換
  \param x	変換されるベクトル
  \param y	変換されるベクトル
  \return	xが変換されたものを下位，yが変換されたものを上位に
		配したベクトル
*/
template <class T, bool MASK=false, class S> inline auto
cvtdown(vec<S> x, vec<S> y)
{
    return cvt<cvt_lower_type<T, S, MASK>, MASK>(x, y);
}

template <class T, bool MASK, size_t N=vec<T>::size, class ITER,
	  std::enable_if_t<iterator_value<ITER>::size == N>* = nullptr>
inline auto
cvtdown(ITER& iter)
{
    return cvtdown<T, MASK>(*iter++);
}

template <class T, bool MASK, size_t N=vec<T>::size, class ITER,
	  std::enable_if_t<(iterator_value<ITER>::size < N)>* = nullptr>
inline auto
cvtdown(ITER& iter)
{
    const auto	x = cvtdown<T, MASK, N/2>(iter);
    const auto	y = cvtdown<T, MASK, N/2>(iter);
    
    return cvtdown<T, MASK>(x, y);
}

template <class T, bool MASK=false, class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
cvtdown(TUPLE&& t)
{
    return tuple_transform([](auto&& x) -> decltype(auto)
			   { return cvtdown<T, MASK, vec<T>::size>(
					std::forward<decltype(x)>(x)); },
			   t);
}

template <class T, bool MASK=false, class... VEC> inline auto
cvtdown(const std::tuple<VEC...>& s, const std::tuple<VEC...>& t)
{
    return tuple_transform([](auto x, auto y)
			   { return cvtdown<T, MASK>(x, y); },
			   s, t);
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/cvt.h"
#elif defined(NEON)
#  include "TU/simd/arm/cvt.h"
#endif

#endif	// !TU_SIMD_CVT_H
