/*
 *  $Id$
 */
#if !defined(__TU_SIMD_SHIFT_H)
#define __TU_SIMD_SHIFT_H

#include "TU/simd/zero.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Elementwise concatinated shift operators				*
************************************************************************/
//! 2つのベクトルを連結した2倍長のベクトルの要素を右シフトした後，下位ベクトルを取り出す．
/*!
  シフト後の上位にはyの要素が入る．
  \param N	シフト数(成分単位), 0 <= N <= vec<T>::size
  \param x	下位のベクトル
  \param y	上位のベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> vec<T>
shift_r(vec<T> x, vec<T> y)
{
    typedef complementary_type<T>	C;

    return cast<T>(shift_r<N>(cast<C>(x), cast<C>(y)));
}

/************************************************************************
*  Elementwise shift operators						*
************************************************************************/
//! ベクトルの要素を左シフトする．
/*!
  シフト後の下位には0が入る．
  \param N	シフト数(成分単位)
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> vec<T>
shift_l(vec<T> x)
{
    return shift_r<vec<T>::size - N>(zero<T>(), x);
}

//! ベクトルの要素を右シフトする．
/*!
  シフト後の上位には0が入る．
  \param N	シフト数(成分単位)
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <size_t N, class T> vec<T>
shift_r(vec<T> x)
{
    return shift_r<N>(x, zero<T>());
}

/************************************************************************
*  Element wise shift to left/right-most				*
************************************************************************/
//! 左端の要素が右端に来るまで右シフトする．
/*!
  シフト後の上位には0が入る．
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <class T> inline vec<T>
shift_lmost_to_rmost(vec<T> x)
{
    return shift_r<vec<T>::size-1>(x);
}

//! 右端の要素が左端に来るまで左シフトする．
/*!
  シフト後の下位には0が入る．
  \param x	シフトされるベクトル
  \return	シフトされたベクトル
*/
template <class T> inline vec<T>
shift_rmost_to_lmost(vec<T> x)
{
    return shift_l<vec<T>::size-1>(x);
}

//! 与えられた値を右端の成分にセットし残りを0としたベクトルを生成する．
/*!
  \param x	セットされる値
  \return	xを右端成分とするベクトル
*/
template <class T> inline vec<T>
set_rmost(T x)
{
    return shift_lmost_to_rmost(vec<T>(x));
}

/************************************************************************
*  Rotation and reverse operators					*
************************************************************************/
//! ベクトルの左回転を行なう．
template <class T> inline vec<T>
rotate_l(vec<T> x)
{
    return shift_r<vec<T>::size-1>(x, x);
}

//! ベクトルの右回転を行なう．
template <class T> inline vec<T>
rotate_r(vec<T> x)
{
    return shift_r<1>(x, x);
}

/************************************************************************
*  Shift operators for vec tuples					*
************************************************************************/
namespace detail
{
  template <size_t N>
  struct generic_shift_l
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x)	const	{ return shift_l<N>(x); }
  };

  template <size_t N>
  struct generic_shift_r
  {
      template <class T_>
      vec<T_>	operator ()(vec<T_> x) const
		{
		    return shift_r<N>(x);
		}
      template <class T_>
      vec<T_>	operator ()(vec<T_> x, vec<T_> y) const
		{
		    return shift_r<N>(x, y);
		}
  };

  struct generic_shift_lmost_to_rmost
  {
      template <class T_> vec<T_>
      operator ()(vec<T_> x)	const	{ return shift_lmost_to_rmost(x); }
  };

  struct generic_shift_rmost_to_lmost
  {
      template <class T_> vec<T_>
      operator ()(vec<T_> x)	const	{ return shift_rmost_to_lmost(x); }
  };
}
    
template <size_t N, class HEAD, class TAIL> inline auto
shift_r(const boost::tuples::cons<HEAD, TAIL>& x,
	const boost::tuples::cons<HEAD, TAIL>& y)
    -> decltype(boost::tuples::cons_transform(x, y,
					      detail::generic_shift_r<N>()))
{
    return boost::tuples::cons_transform(x, y, detail::generic_shift_r<N>());
}

template <size_t N, class HEAD, class TAIL> inline auto
shift_r(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(x, detail::generic_shift_r<N>()))
{
    return boost::tuples::cons_transform(x, detail::generic_shift_r<N>());
}

template <size_t N, class HEAD, class TAIL> inline auto
shift_l(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(x, detail::generic_shift_l<N>()))
{
    return boost::tuples::cons_transform(x, detail::generic_shift_l<N>());
}
    
template <class HEAD, class TAIL> inline auto
shift_lmost_to_rmost(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    x, detail::generic_shift_lmost_to_rmost()))
{
    return boost::tuples::cons_transform(
	       x, detail::generic_shift_lmost_to_rmost());
}
    
template <class HEAD, class TAIL> inline auto
shift_rmost_to_lmost(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(boost::tuples::cons_transform(
		    x, detail::generic_shift_rmost_to_lmost()))
{
    return boost::tuples::cons_transform(
	       x, detail::generic_shift_rmost_to_lmost());
}

}	// namespace simd
}	// namespace TU
#if defined(MMX)
#  include "TU/simd/intel/shift.h"
#elif defined(NEON)
#  include "TU/simd/arm/shift.h"
#endif

#endif	// !__TU_SIMD_ARM_SHIFT_H
