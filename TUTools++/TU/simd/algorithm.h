/*!
  \file		algorithm.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルを対象とした各種アルゴリズムの実装
*/
#ifndef TU_SIMD_ALGORITHM_H
#define TU_SIMD_ALGORITHM_H

#include "TU/simd/arithmetic.h"
#include "TU/simd/load_store_iterator.h"
#include "TU/algorithm.h"
#ifdef TU_DEBUG
#  include <boost/core/demangle.hpp>
#endif

namespace TU
{
namespace simd
{
//! 指定された範囲の各要素に関数を適用する
/*!
  N != 0 の場合，Nで指定した要素数だけ適用し，argは無視．
  N = 0 の場合，ARG = ITERなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または適用要素数
  \param func	適用する関数
*/
template <size_t N, class ITER, class ARG, class FUNC> inline FUNC
for_each(iterator_wrapper<ITER> begin, ARG arg, FUNC func)
{
    constexpr auto	M = make_terminator<ITER>(N);
    
    return TU::for_each<M>(make_accessor(begin), make_terminator<ITER>(arg),
			   func);
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
for_each(iterator_wrapper<ITER0> begin0, ARG arg,
	 iterator_wrapper<ITER1> begin1, FUNC func)
{
    constexpr auto	M = make_terminator<ITER0>(N);

    return TU::for_each<M>(make_accessor(begin0), make_terminator<ITER0>(arg),
			   make_accessor(begin1), func);
}
    
//! 指定された範囲の内積の値を返す
/*!
  N != 0 の場合，Nで指定した要素数の範囲の内積を求め，argは無視．
  N = 0 の場合，ARG = ITER0なら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin0	適用範囲の第1変数の先頭を指す反復子
  \param arg	適用範囲の第1変数の末尾の次を指す反復子または要素数
  \param begin1	適用範囲の第2変数の先頭を指す反復子
  \param init	初期値
  \return	内積の値
*/
template <size_t N, class ITER0, class ARG, class ITER1, class T> inline T
inner_product(iterator_wrapper<ITER0> begin0, ARG arg,
	      iterator_wrapper<ITER1> begin1, T init)
{
#ifdef TU_DEBUG
    std::cout << "(simd)inner_product<" << N << "> ["
	      << print_sizes(range<iterator_wrapper<ITER0>, N>(begin0, arg))
	      << "] ==> ";
#endif
    constexpr auto	M = make_terminator<ITER0>(N);
    
    return hadd(TU::inner_product<M>(make_accessor(begin0),
				     make_terminator<ITER0>(arg),
				     make_accessor(begin1), vec<T>(init)));
}
    
//! 指定された範囲にある要素の2乗和を返す
/*!
  N != 0 の場合，Nで指定した要素数の範囲の2乗和を求め，argは無視．
  N = 0 の場合，ARG = ITERなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または要素数
  \return	2乗和の値
*/
template <size_t N, class ITER, class ARG> inline auto
square(iterator_wrapper<ITER> iter, ARG arg)
{
    constexpr auto	M = make_terminator<ITER>(N);
    
    return hadd(TU::square<M>(make_accessor(iter),
			      make_terminator<ITER>(arg)));
}

template <class FUNC, class ITER> inline auto
make_transform_iterator1(iterator_wrapper<ITER> iter, FUNC func)
{
#ifdef TU_DEBUG
    using	boost::core::demangle;

    std::cout << "(simd)transform_iterator1:\n\t"
	      << demangle(typeid(ITER).name()) << std::endl;
#endif		  
    return wrap_iterator(TU::make_transform_iterator1(make_accessor(iter),
						      func));
}

template <class FUNC, class ITER0, class ITER1> inline auto
make_transform_iterator2(iterator_wrapper<ITER0> begin0,
			 iterator_wrapper<ITER1> begin1, FUNC func)
{
#ifdef TU_DEBUG
    using	boost::core::demangle;

    std::cout << "(simd)transform_iterator2:\n\t"
	      << demangle(typeid(ITER0).name()) << "\n\t"
	      << demangle(typeid(ITER1).name()) << std::endl;
#endif  
    return wrap_iterator(TU::make_transform_iterator2(make_accessor(begin0),
						      make_accessor(begin1),
						      func));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ALGORITHM_H
