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
/************************************************************************
*  iterator_wrapper<ITER>						*
************************************************************************/
//! 反復子をラップして名前空間 TU::simd に取り込むためのクラス
/*!
  本クラスの目的は，以下の2つである．
    (1)	iterator_value<ITER> がSIMDベクトル型となるあらゆる反復子を
	その機能を保持したまま名前空間 TU::simd のメンバとすることにより，
	ADL を発動して simd/algorithm.h に定義された関数を起動できる
	ようにすること．
    (2)	ITER が const T* 型または T* 型のとき，それぞれ load_iterator<T, true>
	store_iterator<T, true> を生成できるようにすること．本クラスでラップ
	されたポインタが指すアドレスは sizeof(vec<T>) にalignされているものと
	みなされる．
	
  \param ITER	ラップする反復子の型
*/ 
template <class ITER>
class iterator_wrapper
    : public boost::iterator_adaptor<iterator_wrapper<ITER>, ITER>
{
  private:
    using	super = boost::iterator_adaptor<iterator_wrapper, ITER>;

  public:
		iterator_wrapper(ITER iter)	:super(iter)	{}
      template <class ITER_,
		std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
		= nullptr>
		iterator_wrapper(ITER_ iter)	:super(iter)	{}
};

//! 反復子をラップして名前空間 TU::simd に取り込む
/*!
  \param iter	ラップする反復子
*/
template <class ITER> inline iterator_wrapper<ITER>
wrap_iterator(ITER iter)
{
    return {iter};
}
    
//! ラップされた反復子からもとの反復子を取り出す
/*!
  \param iter	ラップされた反復子
  \return	もとの反復子
*/
template <class ITER> inline ITER
make_accessor(iterator_wrapper<ITER> iter)
{
    return iter.base();
}

//! ラップされた定数ポインタからSIMDベクトルを読み込む反復子を生成する
/*!
  ラップされたポインタは sizeof(vec<T>) にalignされていなければならない．
  \param p	ラップされた定数ポインタ
  \return	SIMDベクトルを読み込む反復子
*/
template <class T> inline load_iterator<T, true>
make_accessor(iterator_wrapper<const T*> p)
{
    return {p.base()};
}
    
//! ラップされたポインタからSIMDベクトルを書き込む反復子を生成する
/*!
  ラップされたポインタは sizeof(vec<T>) にalignされていなければならない．
  \param p	ラップされたポインタ
  \return	SIMDベクトルを書き込む反復子
*/
template <class T> inline store_iterator<T, true>
make_accessor(iterator_wrapper<T*> p)
{
    return {p.base()};
}
    
namespace detail
{
  template <class ITER>
  struct vsize		// iterator_value<ITER> が vec<T> 型
  {
      constexpr static size_t	value = iterator_value<ITER>::size;
  };
  template <class T>
  struct vsize<T*>
  {
      constexpr static size_t	value = vec<std::remove_cv_t<T> >::size;
  };
}	// namespace detail

//! ある要素型を指定された個数だけカバーするために必要なSIMDベクトルの個数を調べる
/*!
  T を value_type<ITER> がSIMDベクトル型となる場合はその要素型，ITER がポインタの
  場合はそれが指す要素の型としたとき，指定された個数のT型要素をカバーするために
  必要な vec<T> 型SIMDベクトルの最小個数を返す．
  \param n	要素の個数
  \return	nをカバーするために必要なSIMDベクトルの個数
*/
template <class ITER> constexpr inline size_t
make_terminator(size_t n)
{
    return (n ? (n - 1)/detail::vsize<ITER>::value + 1 : 0);
}
    
}	// namespace simd
    
//! 指定された範囲の各要素に関数を適用する
/*!
  N != 0 の場合，Nで指定した要素数だけ適用し，argは無視．
  N = 0 の場合，ARG = ITERなら範囲の末尾の次を，ARG = size_tなら要素数をargで指定，
  \param begin	適用範囲の先頭を指す反復子
  \param arg	適用範囲の末尾の次を指す反復子または適用要素数
  \param func	適用する関数
*/
template <size_t N, class ITER, class ARG, class FUNC> inline FUNC
for_each(simd::iterator_wrapper<ITER> begin, ARG arg, FUNC func)
{
    constexpr auto	M = simd::make_terminator<ITER>(N);
    
    return for_each<M>(simd::make_accessor(begin),
		       simd::make_terminator<ITER>(arg), func);
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
for_each(simd::iterator_wrapper<ITER0> begin0, ARG arg,
	 simd::iterator_wrapper<ITER1> begin1, FUNC func)
{
    constexpr auto	M = simd::make_terminator<ITER0>(N);

    return for_each<M>(simd::make_accessor(begin0),
		       simd::make_terminator<ITER0>(arg),
		       simd::make_accessor(begin1), func);
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
inner_product(simd::iterator_wrapper<ITER0> begin0, ARG arg,
	      simd::iterator_wrapper<ITER1> begin1, T init)
{
#ifdef TU_DEBUG
    std::cout << "(simd)inner_product<" << N << "> ["
	      << print_sizes(range<simd::iterator_wrapper<ITER0>, N>(begin0,
								     arg))
	      << "] ==> ";
#endif
    constexpr auto	M = simd::make_terminator<ITER0>(N);
    
    return simd::hadd(inner_product<M>(simd::make_accessor(begin0),
				       simd::make_terminator<ITER0>(arg),
				       simd::make_accessor(begin1),
				       simd::vec<T>(init)));
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
square(simd::iterator_wrapper<ITER> iter, ARG arg)
{
    constexpr auto	M = simd::make_terminator<ITER>(N);
    
    return simd::hadd(square<M>(simd::make_accessor(iter),
				simd::make_terminator<ITER>(arg)));
}

#ifdef TU_DEBUG
namespace detail
{
  template <class T> inline void
  print_types(std::ostream& out)
  {
      out << '\t' << boost::core::demangle(typeid(T).name()) << std::endl;
  }
  template <class S, class... T> inline std::enable_if_t<sizeof...(T) != 0>
  print_types(std::ostream& out)
  {
      out << '\t' << boost::core::demangle(typeid(S).name()) << std::endl;
      print_types<T...>(out);
  }
}	// namespace detail
#endif
    
template <class FUNC, class... ITER> inline auto
make_transform_iterator(FUNC func, simd::iterator_wrapper<ITER>... iter)
{
#ifdef TU_DEBUG
    std::cout << "(simd)transform_iterator:\n";
    detail::print_types<ITER...>(std::cout);
#endif		  
    return wrap_iterator(make_transform_iterator(func,
						 simd::make_accessor(iter)...));
}
}	// namespace TU
#endif	// !TU_SIMD_ALGORITHM_H
