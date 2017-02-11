/*
 *  $Id$
 */
#ifndef __TU_ALGORITHM_H
#define __TU_ALGORITHM_H

#include <type_traits>	// for std::common_type<TYPES....>

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
  copy(IN ib, IN ie, OUT out, std::integral_constant<size_t, 0>)
  {
      return std::copy(ib, ie, out);
  }
  template <class IN, class OUT, size_t N> inline OUT
  copy(IN ib, IN, OUT out, std::integral_constant<size_t, 1>)
  {
      *out = *ib;
      return ++out;
  }
  template <class IN, class OUT, size_t N> inline OUT
  copy(IN ib, IN ie, OUT out, std::integral_constant<size_t, N>)
  {
      *out = *ib;
      return copy(++ib, ie, ++out, std::integral_constant<size_t, N-1>());
  }

  template <class ITER, class T> inline void
  fill(ITER begin, ITER end, const T& val, std::integral_constant<size_t, 0>)
  {
      std::fill(begin, end, val);
  }
  template <class ITER, class T, size_t N> inline void
  fill(ITER begin, ITER, const T& val, std::integral_constant<size_t, 1>)
  {
      *begin = val;
  }
  template <class ITER, class T, size_t N> inline void
  fill(ITER begin, ITER end, const T& val, std::integral_constant<size_t, N>)
  {
      *begin = val;
      fill(++begin, end, val, std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER, class FUNC> inline FUNC
  for_each(ITER begin, ITER end, FUNC func, std::integral_constant<size_t, 0>)
  {
      return std::for_each(begin, end, func);
  }
  template <class ITER, class FUNC, size_t N> inline FUNC
  for_each(ITER begin, ITER, FUNC func, std::integral_constant<size_t, 1>)
  {
      func(*begin);
      return func;
  }
  template <class ITER, class FUNC, size_t N> inline FUNC
  for_each(ITER begin, ITER end, FUNC func, std::integral_constant<size_t, N>)
  {
      func(*begin);
      return for_each(++begin, end, func,
		      std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER0, class ITER1, class FUNC> inline FUNC
  for_each(ITER0 begin0, ITER0 end0, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, 0>)
  {
      for (; begin0 != end0; ++begin0, ++begin1)
	  func(*begin0, *begin1);
      return func;
  }
  template <class ITER0, class ITER1, class FUNC> inline FUNC
  for_each(ITER0 begin0, ITER0, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, 1>)
  {
      func(*begin0, *begin1);
      return func;
  }
  template <class ITER0, class ITER1, class FUNC, size_t N> inline FUNC
  for_each(ITER0 begin0, ITER0 end0, ITER1 begin1, FUNC func,
	   std::integral_constant<size_t, N>)
  {
      func(*begin0, *begin1);
      return for_each(++begin0, end0, ++begin1, func,
		      std::integral_constant<size_t, N-1>());
  }
    
  template <class ITER0, class ITER1, class T> inline T
  inner_product(ITER0 begin0, ITER0 end0, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 0>)
  {
      return std::inner_product(begin0, end0, begin1, init);
  }
  template <class ITER0, class ITER1, class T> inline T
  inner_product(ITER0 begin0, ITER0, ITER1 begin1, const T& init,
		std::integral_constant<size_t, 1>)
  {
      return init + *begin0 * *begin1;
  }
  template <class ITER0, class ITER1, class T, size_t N> inline T
  inner_product(ITER0 begin0, ITER0 end0, ITER1 begin1, const T& init,
		std::integral_constant<size_t, N>)
  {
      const auto	tmp = init + *begin0 * *begin1;
      return inner_product(++begin0, ++begin1, end0, tmp,
			   std::integral_constant<size_t, N-1>());
  }
}	// namespace detail

template <size_t N, class IN, class OUT> inline OUT
copy(IN ib, IN ie, OUT out)
{
    return detail::copy(ib, ie, out, std::integral_constant<size_t, N>());
}
    
template <size_t N, class ITER, class T> inline void
fill(ITER begin, ITER end, const T& val)
{
    return detail::fill(begin, end, val, std::integral_constant<size_t, N>());
}
    
template <size_t N, class ITER, class FUNC> inline FUNC
for_each(ITER begin, ITER end, FUNC func)
{
    return detail::for_each(begin, end, func,
			    std::integral_constant<size_t, N>());
}
    
template <size_t N, class ITER0, class ITER1, class FUNC> inline FUNC
for_each(ITER0 begin0, ITER0 end0, ITER1 begin1, FUNC func)
{
    return detail::for_each(begin0, end0, begin1, func,
			    std::integral_constant<size_t, N>());
}
    
template <size_t N, class ITER0, class ITER1, class T> inline T
inner_product(ITER0 begin0, ITER0 end0, ITER1 begin1, const T& init)
{
    return detail::inner_product(begin0, end0, begin1, init,
				 std::integral_constant<size_t, N>());
}
    
}	// namespace TU
#endif	// !__TU_ALGORITHM_H
