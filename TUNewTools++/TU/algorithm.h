/*
 *  $Id$
 */
/*!
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#ifndef __TU_ALGORITHM_H
#define __TU_ALGORITHM_H

#include <type_traits>		// for std::common_type<TYPES....>
#include <algorithm>

namespace TU
{
//! 与えられた二つの整数の最大公約数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \return	mとnの最大公約数
*/
template <class S, class T> constexpr typename std::common_type<S, T>::type
gcd(S m, T n)
{
    return (n == 0 ? m : gcd(n, m % n));
}

//! 与えられた三つ以上の整数の最大公約数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \param args	第3, 第4,...の整数
  \return	m, n, args...の最大公約数
*/
template <class S, class T, class... ARGS>
constexpr typename std::common_type<S, T, ARGS...>::type
gcd(S m, T n, ARGS... args)
{
    return gcd(gcd(m, n), args...);
}

//! 与えられた二つの整数の最小公倍数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \return	mとnの最小公倍数
*/
template <class S, class T> constexpr typename std::common_type<S, T>::type
lcm(S m, T n)
{
    return (m*n == 0 ? 0 : (m / gcd(m, n)) * n);
}

//! 与えられた三つ以上の整数の最小公倍数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \param args	第3, 第4,...の整数
  \return	m, n, args...の最小公倍数
*/
template <class S, class T, class... ARGS>
constexpr typename std::common_type<S, T, ARGS...>::type
lcm(S m, T n, ARGS... args)
{
    return lcm(lcm(m, n), args...);
}

namespace detail
{
  template <class IN, class OUT> inline OUT
  copy(IN in, IN ie, OUT out, std::integral_constant<size_t, 0>)
  {
      return std::copy(in, ie, out);
  }
  template <class IN, class OUT> inline OUT
  copy(IN in, size_t n, OUT out, std::integral_constant<size_t, 0>)
  {
      return std::copy_n(in, n, out);
  }
  template <class IN, class ARG, class OUT> inline OUT
  copy(IN in, ARG, OUT out, std::integral_constant<size_t, 1>)
  {
      *out = *in;
      return ++out;
  }
  template <class IN, class ARG, class OUT, size_t N> inline OUT
  copy(IN in, ARG arg, OUT out, std::integral_constant<size_t, N>)
  {
      *out = *in;
      return copy(++in, arg, ++out, std::integral_constant<size_t, N-1>());
  }

  template <class ITER, class T> inline void
  fill(ITER begin, ITER end, const T& val, std::integral_constant<size_t, 0>)
  {
      std::fill(begin, end, val);
  }
  template <class ITER, class T> inline void
  fill(ITER begin, size_t n, const T& val, std::integral_constant<size_t, 0>)
  {
      std::fill_n(begin, n, val);
  }
  template <class ITER, class T, size_t N> inline void
  fill(ITER begin, ITER, const T& val, std::integral_constant<size_t, 1>)
  {
      *begin = val;
  }
  template <class ITER, class ARG, class T, size_t N> inline void
  fill(ITER begin, ARG arg, const T& val, std::integral_constant<size_t, N>)
  {
      *begin = val;
      fill(++begin, arg, val, std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER0, class ITER1, class T> inline T
  inner_product(ITER0 begin0, ITER0 end0, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 0>)
  {
      return std::inner_product(begin0, end0, begin1, init);
  }
  template <class ITER0, class ITER1, class T> inline T
  inner_product(ITER0 begin0, size_t n, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 0>)
  {
      auto	val = init;
      for (size_t i = 0; i != n; ++i, ++begin0, ++begin1)
	  val += *begin0 * *begin1;
      return val;
  }
  template <class ITER0, class ARG, class ITER1, class T> inline T
  inner_product(ITER0 begin0, ARG, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 1>)
  {
      return init + *begin0 * *begin1;
  }
  template <class ITER0, class ARG, class ITER1, class T, size_t N> inline T
  inner_product(ITER0 begin0, ARG arg, ITER1 begin1, const T& init,
		std::integral_constant<size_t, N>)
  {
      const T	tmp = init + *begin0 * *begin1;
      return inner_product(++begin0, arg, ++begin1, tmp,
			   std::integral_constant<size_t, N-1>());
  }

  template <class T>
  inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
  square(const T& val)
  {
      return val * val;
  }
  template <class ITER> inline auto
  square(ITER begin, ITER end, std::integral_constant<size_t, 0>)
  {
      using value_type	= typename std::iterator_traits<ITER>::value_type;
    
      value_type	val = 0;
      for (; begin != end; ++begin)
	  val += square(*begin);
      return val;
  }
  template <class ITER> inline auto
  square(ITER begin, size_t n, std::integral_constant<size_t, 0>)
  {
      using value_type	= typename std::iterator_traits<ITER>::value_type;

      value_type	val = 0;
      for (size_t i = 0; i != n; ++i, ++begin)
	  val += square(*begin);
      return val;
  }
  template <class ITER, class ARG> inline auto
  square(ITER begin, ARG, std::integral_constant<size_t, 1>)
  {
      return square(*begin);
  }
  template <class ITER, class ARG, size_t N> inline auto
  square(ITER begin, ARG arg, std::integral_constant<size_t, N>)
  {
      const auto	tmp = square(*begin);
      return tmp + square(++begin, arg, std::integral_constant<size_t, N-1>());
  }
}	// namespace detail

//! 指定された範囲をコピーする
/*!
  N != 0 の場合，Nで指定した要素数をコピーし，argは無視．
  N = 0 の場合，ARG = INならコピー元の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param in	コピー元の先頭を指す反復子
  \param arg	コピー元の末尾の次を指す反復子またはコピーする要素数
  \param out	コピー先の先頭を指す反復子
  \return	コピー先の末尾の次
*/
template <size_t N, class IN, class ARG, class OUT> inline OUT
copy(IN in, ARG arg, OUT out)
{
    return detail::copy(in, arg, out, std::integral_constant<size_t, N>());
}
    
//! 指定された範囲を与えられた値で埋める
/*!
  N != 0 の場合，Nで指定した要素数だけ埋め，argは無視．
  N = 0 の場合，ARG = INなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	埋める範囲の先頭を指す反復子
  \param arg	埋める範囲の末尾の次を指す反復子または埋める要素数
  \param val	埋める値
*/
template <size_t N, class ITER, class ARG, class T> inline void
fill(ITER begin, ARG arg, const T& val)
{
    return detail::fill(begin, arg, val, std::integral_constant<size_t, N>());
}
    
//! 指定された範囲の内積の値を返す
/*!
  N != 0 の場合，Nで指定した要素数の範囲の内積を求め，argは無視．
  N = 0 の場合，ARG = INなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin0	適用範囲の第1変数の先頭を指す反復子
  \param arg	適用範囲の第1変数の末尾の次を指す反復子または要素数
  \param begin1	適用範囲の第2変数の先頭を指す反復子
  \param init	初期値
  \return	内積の値
*/
template <size_t N, class ITER0, class ARG, class ITER1, class T> inline T
inner_product(ITER0 begin0, ARG arg, ITER1 begin1, const T& init)
{
    return detail::inner_product(begin0, arg, begin1, init,
				 std::integral_constant<size_t, N>());
}
    
//! 指定された範囲にある要素の2乗和を返す
/*!
  N != 0 の場合，Nで指定した要素数の範囲の2乗和を求め，argは無視．
  N = 0 の場合，ARG = INなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または要素数
  \return	2乗和の値
*/
template <size_t N, class ITER, class ARG>
inline typename std::iterator_traits<ITER>::value_type
square(ITER begin, ARG arg)
{
    return detail::square(begin, arg, std::integral_constant<size_t, N>());
}
    
}	// namespace TU
#endif	// !__TU_ALGORITHM_H
