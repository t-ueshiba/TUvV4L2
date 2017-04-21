/*!
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#ifndef __TU_ALGORITHM_H
#define __TU_ALGORITHM_H

#include <cstddef>		// for size_t
#include <cmath>		// for std::sqrt()
#include <iterator>		// for std::iterator_traits<ITER>
#include <type_traits>		// for std::common_type<TYPES....>
#include <algorithm>		// for std::copy(), std::copy_n(),...
#include <numeric>		// for std::inner_product()
#ifdef TU_DEBUG
#  include <iostream>
#endif

namespace TU
{
#ifdef TU_DEBUG
template <class ITER, size_t SIZE>	class range;
template <class E>			class sizes_holder;

template <class E>
sizes_holder<E>	print_sizes(const E& expr);
template <class E>
std::ostream&	operator <<(std::ostream& out, const sizes_holder<E>& holder);
#endif

/************************************************************************
*  generic algorithms							*
************************************************************************/
//! 条件を満たす要素が前半に，そうでないものが後半になるように並べ替える．
/*!
  \param begin	データ列の先頭を示す反復子
  \param end	データ列の末尾を示す反復子
  \param pred	条件を指定する単項演算子
  \return	条件を満たさない要素の先頭を示す反復子
*/
template <class Iter, class Pred> Iter
pull_if(Iter begin, Iter end, Pred pred)
{
    for (Iter iter = begin; iter != end; ++iter)
	if (pred(*iter))
	    std::iter_swap(begin++, iter);
    return begin;
}

//! 2つの引数の差の絶対値を返す．
template <class T> inline T
diff(const T& a, const T& b)
{
    return (a > b ? a - b : b - a);
}

//! 与えられた二つの整数の最大公約数を求める．
/*!
  \param m	第1の整数
  \param n	第2の整数
  \return	mとnの最大公約数
*/
template <class S, class T> constexpr std::common_type_t<S, T>
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
constexpr std::common_type_t<S, T, ARGS...>
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
template <class S, class T> constexpr std::common_type_t<S, T>
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
constexpr std::common_type_t<S, T, ARGS...>
lcm(S m, T n, ARGS... args)
{
    return lcm(lcm(m, n), args...);
}

namespace detail
{
  template <class ITER, class FUNC> inline FUNC
  for_each(ITER begin, ITER end, FUNC func, std::integral_constant<size_t, 0>)
  {
      return std::for_each(begin, end, func);
  }
  template <class ITER, class FUNC> inline FUNC
  for_each(ITER begin, size_t n, FUNC func, std::integral_constant<size_t, 0>)
  {
      return std::for_each(begin, begin + n, func);
  }
  template <class ITER, class ARG, class FUNC> inline FUNC
  for_each(ITER begin, ARG, FUNC func, std::integral_constant<size_t, 1>)
  {
      func(*begin);
      return std::move(func);
  }
  template <class ITER, class ARG, class FUNC, size_t N> inline FUNC
  for_each(ITER begin, ARG arg, FUNC func, std::integral_constant<size_t, N>)
  {
      func(*begin);
      return for_each(++begin, arg, func,
		      std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER0, class ITER1, class FUNC> inline FUNC
  for_each(ITER0 begin0, ITER0 end0, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, 0>)
  {
      for (; begin0 != end0; ++begin0, ++begin1)
	  func(*begin0, *begin1);
      return std::move(func);
  }
  template <class ITER0, class ITER1, class FUNC> inline FUNC
  for_each(ITER0 begin0, size_t n, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, 0>)
  {
      for (; n--; ++begin0, ++begin1)
	  func(*begin0, *begin1);
      return std::move(func);
  }
  template <class ITER0, class ARG, class ITER1, class FUNC> inline FUNC
  for_each(ITER0 begin0, ARG, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, 1>)
  {
      func(*begin0, *begin1);
      return std::move(func);
  }
  template <class ITER0, class ARG, class ITER1, class FUNC, size_t N>
  inline FUNC
  for_each(ITER0 begin0, ARG arg, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, N>)
  {
      func(*begin0, *begin1);
      return for_each(++begin0, arg, ++begin1, func,
		      std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER0, class ITER1, class T> inline T
  inner_product(ITER0 begin0, ITER0 end0, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 0>)
  {
      auto	val = init;
      for (; begin0 != end0; ++begin0, ++begin1)
	  val += *begin0 * *begin1;
      return val;
  }
  template <class ITER0, class ITER1, class T> T
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
      return inner_product(begin0 + 1, arg, begin1 + 1,
			   init + *begin0 * *begin1,
			   std::integral_constant<size_t, N-1>());
  }

  template <class T> inline std::enable_if_t<std::is_arithmetic<T>::value, T>
  square(const T& val)
  {
      return val * val;
  }
  template <class ITER> auto
  square(ITER begin, ITER end, std::integral_constant<size_t, 0>)
  {
      using value_type	= typename std::iterator_traits<ITER>::value_type;
    
      value_type	val = 0;
      for (; begin != end; ++begin)
	  val += square(*begin);
      return val;
  }
  template <class ITER> auto
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

//! 指定された範囲の各要素に関数を適用する
/*!
  N != 0 の場合，Nで指定した要素数だけ適用し，argは無視．
  N = 0 の場合，ARG = ITERなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または適用要素数
  \param func	適用する関数
*/
template <size_t N, class ITER, class ARG, class FUNC> inline FUNC
for_each(ITER begin, ARG arg, FUNC func)
{
    return detail::for_each(begin, arg, func,
			    std::integral_constant<size_t, N>());
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
    for_each<N>(begin, arg, [&val](auto&& dst){ dst = val; });
}
    
//! 指定された2つの範囲の各要素に2変数関数を適用する
/*!
  N != 0 の場合，Nで指定した要素数だけ適用し，argは無視．
  N = 0 の場合，ARG = ITER0なら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin0	第1の適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または適用要素数
  \param begin1	第2の適用範囲の先頭を指す反復子
  \param func	適用する関数
*/
template <size_t N, class ITER0, class ARG, class ITER1, class FUNC> inline FUNC
for_each(ITER0 begin0, ARG arg, ITER1 begin1, FUNC func)
{
    return detail::for_each(begin0, arg, begin1, func,
			    std::integral_constant<size_t, N>());
}
    
//! 指定された範囲をコピーする
/*!
  N != 0 の場合，Nで指定した要素数をコピーし，argは無視．
  N = 0 の場合，ARG = INならコピー元の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param in	コピー元の先頭を指す反復子
  \param arg	コピー元の末尾の次を指す反復子またはコピーする要素数
  \param out	コピー先の先頭を指す反復子
  \return	コピー先の末尾の次
*/
template <size_t N, class IN, class ARG, class OUT> inline void
copy(IN in, ARG arg, OUT out)
{
#ifdef TU_DEBUG
    std::cout << "copy<" << N << "> ["
	      << print_sizes(range<IN, N>(in, arg)) << ']' << std::endl;
#endif
    for_each<N>(in, arg, out, [](const auto& src, auto&& dst){ dst = src; });
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
#ifdef TU_DEBUG
    std::cout << "inner_product<" << N << "> ["
	      << print_sizes(range<ITER0, N>(begin0, arg)) << ']' << std::endl;
#endif
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
template <size_t N, class ITER, class ARG> inline auto
square(ITER begin, ARG arg)
{
    return detail::square(begin, arg, std::integral_constant<size_t, N>());
}

template <class T> inline std::enable_if_t<std::is_arithmetic<T>::value, T>
square(const T& val)
{
    return detail::square(val);
}
    
}	// namespace TU
#endif	// !__TU_ALGORITHM_H
