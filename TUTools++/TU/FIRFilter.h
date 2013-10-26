/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: FIRFilter.h 1425 2013-09-28 07:12:18Z ueshiba $
 */
/*!
  \file		FIRFilter.h
  \brief	一般的なfinite impulse response filterを表すクラスの定義と実装
*/
#ifndef	__TUFIRFilter_h
#define	__TUFIRFilter_h

#include <algorithm>
#include <boost/array.hpp>
#include "TU/iterator.h"
#include "TU/SeparableFilter2.h"

namespace TU
{
/************************************************************************
*  class FIRFilter<D, T>						*
************************************************************************/
//! 片側Infinite Inpulse Response Filterを表すクラス
template <size_t D, class T=float>
class FIRFilter
{
  public:
    typedef T				coeff_type;
    typedef boost::array<T, D>		coeffs_type;

    FIRFilter&		initialize(const T c[D])			;
    void		limits(T& limit0, T& limit1, T& limit2)	const	;
    template <class IN, class OUT>
    OUT			convolve(IN ib, IN ie, OUT out)		const	;
    
    const coeffs_type&	c()				const	{return _c;}
    static size_t	winSize()				{return D;}
    static size_t	outLength(size_t inLength)		;
	
  private:
    coeffs_type	_c;	//!< フィルタ係数
};

//! フィルタのz変換係数をセットする
/*!
  \param c	z変換係数. z変換関数は
		\f[
		  H(z^{-1}) = {c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots + c_{0}z^{-(D-1)}
		\f]
  \return	このフィルタ自身
*/
template <size_t D, class T> FIRFilter<D, T>&
FIRFilter<D, T>::initialize(const T c[D])
{
    std::copy(c, c + D,	_c.begin());

    return *this;
}

//! 特定の入力データ列に対してフィルタを適用した場合の極限値を求める
/*!
  \param limit0		一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1		傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2		2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <size_t D, class T> void
FIRFilter<D, T>::limits(T& limit0, T& limit1, T& limit2) const
{
    T	x0 = 0, x1 = 0, x2 = 0;
    for (size_t i = 0; i < D; ++i)
    {
	x0 +=	      _c[i];
	x1 +=	    i*_c[D-1-i];
	x2 += (i-1)*i*_c[D-1-i];
    }
    limit0 =  x0;
    limit1 = -x1;
    limit2 =  x1 + x2;
}

//! フィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> OUT
FIRFilter<D, T>::convolve(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    typedef typename coeffs_type::const_iterator		citerator;
    
    return std::copy(make_fir_filter_iterator<D, value_type>(ib, _c.begin()),
		     make_fir_filter_iterator<D, value_type, citerator>(ie),
		     out);
}

//! 与えられた長さの入力データ列に対する出力データ列の長さを返す
/*!
  \param inLength	入力データ列の長さ
  \return		出力データ列の長さ
*/
template <size_t D, class T> inline size_t
FIRFilter<D, T>::outLength(size_t inLength)
{
    return inLength + 1 - D;
}

/************************************************************************
*  class FIRFilter2<D, T>						*
************************************************************************/
//! 2次元Finite Inpulse Response Filterを表すクラス
template <size_t D, class T=float>
class FIRFilter2 : public SeparableFilter2<FIRFilter<D, T> >
{
  private:
    typedef FIRFilter<D, T>			fir_type;
    typedef SeparableFilter2<fir_type>		super;

  public:
    typedef typename fir_type::coeff_type	coeff_type;
    typedef typename fir_type::coeffs_type	coeffs_type;

  public:
    FIRFilter2&		initialize(const T cH[], const T cV[])		;

    template <class IN, class OUT>
    void		convolve(IN ib, IN iue, OUT out)	const	;
    using		super::filterH;
    using		super::filterV;

    const coeffs_type&	cH()		const	{return filterH().c();}
    const coeffs_type&	cV()		const	{return filterV().c();}
    static size_t	winSize()
			{
			    return fir_type::winSize();
			}
    static size_t	outLength(size_t inLen)
			{
			    return fir_type::outLength(inLen);
			}
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cH	横方向z変換係数
  \param cV	縦方向z変換係数
  \return	このフィルタ自身
*/
template <size_t D, class T> inline FIRFilter2<D, T>&
FIRFilter2<D, T>::initialize(const T cH[], const T cV[])
{
    filterH().initialize(cH);
    filterV().initialize(cV);

    return *this;
}

template <size_t D, class T> template <class IN, class OUT> inline void
FIRFilter2<D, T>::convolve(IN ib, IN ie, OUT out) const
{
    std::advance(out, D/2);
    super::convolve(ib, ie, make_row_iterator(out, D/2));
}

}
#endif	/* !__TUFIRFilter_h */
