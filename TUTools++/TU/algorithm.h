/*!
  \mainpage	libTUTools++ - 配列，ベクトル，行列，画像等の基本的なデータ型とそれに付随したアルゴリズムを収めたライブラリ
  \anchor	libTUTools
  \author	Toshio UESHIBA
  \copyright	2002-2017年 植芝俊夫

  \section abstract 概要
  libTUTools++は，配列，ベクトル，行列，画像等の基本的なデータ型とそれ
  に付随したアルゴリズムを収めたライブラリである．現在実装されている主
  要なクラスおよび関数はおおまかに以下の分野に分類される．

  <b>多次元配列</b>
  - #TU::array
  - #TU::Array
  - #TU::Array2

  <b>ベクトルと行列および線形計算</b>
  - #TU::Vector
  - #TU::Matrix
  - #TU::LUDecomposition
  - #TU::Householder
  - #TU::QRDecomposition
  - #TU::TriDiagonal
  - #TU::BiDiagonal
  - #TU::SVDecomposition
  - #TU::BlockDiagonalMatrix
  - #TU::SparseMatrix
  - #TU::BandMatrix

  <b>非線形最適化</b>
  - #TU::NullConstraint
  - #TU::ConstNormConstraint
  - #TU::minimizeSquare(const F&, const G&, AT&, size_t, double)
  - #TU::minimizeSquareSparse(const F&, const G&, ATA&, IB, IB, size_t, double)

  <b>RANSAC</b>
  - #TU::ransac(const PointSet&, Model&, Conform, double)

  <b>グラフカット</b>
  - #boost::GraphCuts

  <b>動的計画法</b>
  - #TU::DP

  <b>点，直線，平面等の幾何要素とその変換</b>
  - #TU::Point2
  - #TU::Point3
  - #TU::HyperPlane
  - #TU::Normalize
  - #TU::Projectivity
  - #TU::Affinity
  - #TU::Rigidity
  - #TU::BoundingBox
  
  <b>投影の幾何</b>
  - #TU::IntrinsicBase
  - #TU::IntrinsicWithFocalLength
  - #TU::IntrinsicWithEuclideanImagePlane
  - #TU::Intrinsic
  - #TU::IntrinsicWithDistortion
  - #TU::CanonicalCamera
  - #TU::Camera
  
  <b>画素と画像</b>
  - #TU::RGB
  - #TU::BGR
  - #TU::RGBA
  - #TU::ABGR
  - #TU::ARGB
  - #TU::BGRA
  - #TU::YUV444
  - #TU::YUV422
  - #TU::YUV411
  - #TU::Image
  - #TU::GenericImage
  - #TU::ComplexImage
  - #TU::CCSImage
  - #TU::Movie

  <b>画像処理</b>
  - #TU::Rectify
  - #TU::EdgeDetector
  - #TU::IntegralImage
  - #TU::DiagonalIntegralImage
  
  <b>画像に限らない信号処理</b>
  - #TU::Warp
  - #TU::Filter2
  - #TU::SeparableFilter2
  - #TU::IIRFilter
  - #TU::BidirectionalIIRFilter
  - #TU::BidirectionalIIRFilter2
  - #TU::DericheConvolver
  - #TU::DericheConvolver2
  - #TU::GaussianConvolver
  - #TU::GaussianConvolver2
  - #TU::BoxFilter
  - #TU::BoxFilter2
  - #TU::GuidedFilter
  - #TU::GuidedFilter2
  - #TU::FIRFilter
  - #TU::FIRGaussianConvolver
  - #TU::WeightedMedianFilter
  - #TU::TreeFilter

  <b>特殊データ構造</b>
  - #TU::List
  - #TU::PSTree
  - #TU::NDTree
  
  <b>Bezier曲線とBezier曲面</b>
  - #TU::BezierCurve
  - #TU::BezierSurface
  
  <b>B-spline曲線とB-spline曲面</b>
  - #TU::BSplineKnots
  - #TU::BSplineCurve
  - #TU::BSplineSurface
  
  <b>メッシュ</b>
  - #TU::Mesh

  <b>アルゴリズム</b>
  - #TU::pull_if(Iter, Iter, Pred)
  - #TU::diff(const T&, const T&)
  - #TU::op3x3(Iterator begin, Iterator end, OP op)
  - #TU::max3x3(P p, P q, P r)
  - #TU::min3x3(P p, P q, P r)
  - #TU::mopOpen(Iterator begin, Iterator end, size_t niter)
  - #TU::mopClose(Iterator begin, Iterator end, size_t niter)

  <b>関数オブジェクト</b>
  - #TU::identity
  - #TU::plus_assign
  - #TU::minus_assign
  - #TU::multiplies_assign
  - #TU::divides_assign
  - #TU::modulus_assign
  - #TU::bit_and_assign
  - #TU::bit_or_assign
  - #TU::bit_xor_assign
  - #TU::square()
  - #TU::length()
  - #TU::square_distance()
  - #TU::distance()
  - #TU::gcd()
  - #TU::lcm()

  <b>反復子</b>
  - #TU::make_mbr_iterator()
  - #TU::make_first_iterator()
  - #TU::make_second_iterator()
  - #TU::zip_iterator
  - #TU::assignment_iterator
  - #TU::row2col
  - #TU::column_iterator
  - #TU::ring_iterator
  - #TU::box_filter_iterator
  - #TU::iir_filter_iterator
  
  <b>マニピュレータ</b>
  - #TU::skipl(std::istream&)
  - #TU::IOManip
  - #TU::IManip1
  - #TU::OManip1
  - #TU::IOManip1
  - #TU::IManip2
  - #TU::OManip2

  <b>ストリーム
  - #TU::fdistream
  - #TU::fdostream
  - #TU::fdstream
  
  <b>シリアルインタフェース</b>
  - #TU::Serial
  - #TU::TriggerGenerator
  - #TU::PM16C_04
  - #TU::SHOT602

  <b>SIMD命令</b>
  - #TU::simd::vec
  
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#ifndef TU_ALGORITHM_H
#define TU_ALGORITHM_H

#include <cmath>		// for std::sqrt()
#include <iterator>		// for std::iterator_traits<ITER>
#include <type_traits>		// for std::common_type<TYPES....>
#include <algorithm>		// for std::copy(), std::copy_n(),...
#include <numeric>		// for std::inner_product()
#include <utility>		// for std::forward()
#ifdef TU_DEBUG
#  include <iostream>
#endif

namespace std
{
#if __cplusplus < 201700L
namespace detail
{
  template <class IN, class OUT, class SIZE, class GEN> OUT
  sample(IN in, IN ie, input_iterator_tag, OUT out, random_access_iterator_tag,
	 SIZE n, GEN&& gen)
  {
      using distrib_type = std::uniform_int_distribution<SIZE>;
      using param_type   = typename distrib_type::param_type;

      distrib_type	distrib{};
      SIZE		nsampled = 0;

    // 最初のn個をコピー
      for (; in != ie && nsampled != n; ++in, ++nsampled)
	  out[nsampled] = *in;

      for (auto ninputs = nsampled; in != ie; ++in, ++ninputs)
      {
	  const auto	i = distrib(gen, param_type{0, ninputs});
	  if (i < n)
	      out[i] = *in;
      }

      return out + nsampled;
  }

  template<class IN, class OUT, class CAT, class SIZE, class GEN> OUT
  sample(IN in, IN ie, forward_iterator_tag, OUT out, CAT, SIZE n, GEN&& gen)
  {
      using distrib_type = std::uniform_int_distribution<SIZE>;
      using param_type	 = typename distrib_type::param_type;

      distrib_type	distrib{};
      SIZE		nunsampled = std::distance(in, ie);

      if (n > nunsampled)
	  n = nunsampled;
      
      for (; n != 0; ++in)
	  if (distrib(gen, param_type{0, --nunsampled}) < n)
	  {
	      *out++ = *in;
	      --n;
	  }

      return out;
  }
}	// namespace detail
    
/// Take a random sample from a population.
template<class IN, class OUT, class SIZE, class GEN> OUT
sample(IN in, IN ie, OUT out, SIZE n, GEN&& gen)
{
    using in_cat  = typename std::iterator_traits<IN>::iterator_category;
    using out_cat = typename std::iterator_traits<OUT>::iterator_category;

    return detail::sample(in, ie, in_cat{}, out, out_cat{},
			  n, std::forward<GEN>(gen));
}
#endif    
}	// namespace std

//! libTUTools++ のクラスや関数等を収める名前空間
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

/************************************************************************
*  for_each<N>(ITER begin, ARG arg, FUNC func)				*
************************************************************************/
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
      return for_each(++begin, arg, func, std::integral_constant<size_t, N-1>());
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
    
/************************************************************************
*  fill<N>(ITER begin, ARG arg, const T& val)				*
************************************************************************/
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
    for_each<N>(begin, arg,
		[&val](auto&& dst){ std::forward<decltype(dst)>(dst) = val; });
}
    
/************************************************************************
*  for_each<N>(ITER0 begin, ARG arg, ITER1 BEGIN1, FUNC func)		*
************************************************************************/
namespace detail
{
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
}	// namespace detail
    
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
    
/************************************************************************
*  copy<N>(IN in, ARG arg, OUT out)					*
************************************************************************/
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
    for_each<N>(in, arg, out,
		[](auto&& src, auto&& dst)
		{ std::forward<decltype(dst)>(dst)
			= std::forward<decltype(src)>(src); });
}

/************************************************************************
*  inner_product<N>(ITER0 begin0, ARG arg, ITER1 begin1, const T& init)	*
************************************************************************/
namespace detail
{
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
}	// namespace detail

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
    
/************************************************************************
*  square<N>(ITER begin, ARG arg)					*
************************************************************************/
namespace detail
{
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

//! 与えられた数値の2乘値を返す
/*
  \param val	数値
  \retrun	2乘値
*/ 
template <class T> inline std::enable_if_t<std::is_arithmetic<T>::value, T>
square(const T& val)
{
    return detail::square(val);
}
    
}	// namespace TU
#endif	// !TU_ALGORITHM_H
