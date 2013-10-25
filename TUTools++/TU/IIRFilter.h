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
 *  $Id$
 */
/*!
  \file		IIRFilter.h
  \brief	各種infinite impulse response filterに関するクラスの定義と実装
*/
#ifndef	__TUIIRFilterPP_h
#define	__TUIIRFilterPP_h

#include <algorithm>
#include <boost/array.hpp>
#include "TU/SeparableFilter2.h"

namespace TU
{
/************************************************************************
*  class IIRFilter<D, T>						*
************************************************************************/
//! 片側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float> class IIRFilter
{
  public:
    typedef T				coeff_type;
    typedef boost::array<T, D>		coeffs_type;
    
    IIRFilter&	initialize(const T c[D+D])				;
    void	limitsF(T& limit0F, T& limit1F, T& limit2F)	const	;
    void	limitsB(T& limit0B, T& limit1B, T& limit2B)	const	;
    template <class IN, class OUT>
    OUT		forward(IN ib, IN ie, OUT out)			const	;
    template <class IN, class OUT>
    OUT		backward(IN ib, IN ie, OUT out)			const	;
    
    const coeffs_type&	ci()			const	{return _ci;}
    const coeffs_type&	co()			const	{return _co;}
	
    static u_int	outLength(u_int inLength)	;

  private:
    coeffs_type	_ci;	//!< 入力フィルタ係数
    coeffs_type	_co;	//!< 出力フィルタ係数
};

//! フィルタのz変換係数をセットする
/*!
  \param c	z変換係数. z変換関数は，前進フィルタの場合は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
		後退フィルタの場合は
		\f[
		  H(z) = \frac{c_{0}z + c_{1}z^2 + \cdots + c_{D-1}z^D}
		       {1 - c_{D}z - c_{D+1}z^2 - \cdots - c_{2D-1}z^D}
		\f]
  \return	このフィルタ自身
*/
template <u_int D, class T> IIRFilter<D, T>&
IIRFilter<D, T>::initialize(const T c[D+D])
{
    std::copy(c,     c + D,	_ci.begin());
    std::copy(c + D, c + D + D,	_co.begin());

    return *this;
}

//! 特定の入力データ列に対して前進方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0F	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1F	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2F	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
IIRFilter<D, T>::limitsF(T& limit0F, T& limit1F, T& limit2F) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      _ci[i];
	d0 -=	      _co[i];
	n1 +=	    i*_ci[D-1-i];
	d1 -=	(i+1)*_co[D-1-i];
	n2 += (i-1)*i*_ci[D-1-i];
	d2 -= i*(i+1)*_co[D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0F =  x0;
    limit1F = -x1;
    limit2F =  x1 + x2;
}

//! 特定の入力データ列に対して後退方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0B	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1B	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2B	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
IIRFilter<D, T>::limitsB(T& limit0B, T& limit1B, T& limit2B) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      _ci[i];
	d0 -=	      _co[i];
	n1 +=	(i+1)*_ci[i];
	d1 -=	(i+1)*_co[i];
	n2 += i*(i+1)*_ci[i];
	d2 -= i*(i+1)*_co[i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0B = x0;
    limit1B = x1;
    limit2B = x1 + x2;
}

//! 前進方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::forward(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    
    return std::copy(make_iir_filter_iterator<D, true, value_type>(
			 ib, _ci.begin(), _co.begin()),
		     make_iir_filter_iterator<D, true, value_type>(
			 ie, _ci.begin(), _co.begin()),
		     out);
}
    
//! 後退方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param oe	出力データ列の末尾の次を指す反復子
  \return	出力データ列の先頭を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::backward(IN ib, IN ie, OUT oe) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    
    return std::copy(make_iir_filter_iterator<D, false, value_type>(
			 ib, _ci.rbegin(), _co.rbegin()),
		     make_iir_filter_iterator<D, false, value_type>(
			 ie, _ci.rbegin(), _co.rbegin()),
		     oe);
}

//! 与えられた長さの入力データ列に対する出力データ列の長さを返す
/*!
  \param inLength	入力データ列の長さ
  \return		出力データ列の長さ
*/
template <u_int D, class T> inline u_int
IIRFilter<D, T>::outLength(u_int inLength)
{
    return inLength;
}
    
/************************************************************************
*  class BidirectionalIIRFilter<D, T>					*
************************************************************************/
//! 両側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float> class BidirectionalIIRFilter
{
  private:
    typedef IIRFilter<D, T>			iirf_type;
    typedef Array<u_char>			buf_type;
    
  public:
    typedef typename iirf_type::coeff_type	coeff_type;
    typedef typename iirf_type::coeffs_type	coeffs_type;
    
  //! 微分の階数
    enum Order
    {
	Zeroth,		//!< 0階微分
	First,		//!< 1階微分
	Second		//!< 2階微分
    };

    BidirectionalIIRFilter&
		initialize(const T cF[D+D], const T cB[D+D])		;
    BidirectionalIIRFilter&
		initialize(const T c[D+D], Order order)			;
    void	limits(T& limit0, T& limit1, T& limit2)		const	;
    template <class IN, class OUT>
    OUT		convolve(IN ib, IN ie, OUT out)			const	;

    const coeffs_type&	ciF()			const	{return _iirF.ci();}
    const coeffs_type&	coF()			const	{return _iirF.co();}
    const coeffs_type&	ciB()			const	{return _iirB.ci();}
    const coeffs_type&	coB()			const	{return _iirB.co();}

    static u_int	outLength(u_int inLength)	;
	
  private:
    IIRFilter<D, T>	_iirF;
    IIRFilter<D, T>	_iirB;
};

//! フィルタのz変換係数をセットする
/*!
  \param cF	前進z変換係数. z変換は 
		\f[
		  H^F(z^{-1}) = \frac{c^F_{D-1} + c^F_{D-2}z^{-1}
		  + c^F_{D-3}z^{-2} + \cdots
		  + c^F_{0}z^{-(D-1)}}{1 - c^F_{2D-1}z^{-1}
		  - c^F_{2D-2}z^{-2} - \cdots - c^F_{D}z^{-D}}
		\f]
		となる. 
  \param cB	後退z変換係数. z変換は
		\f[
		  H^B(z) = \frac{c^B_{0}z + c^B_{1}z^2 + \cdots + c^B_{D-1}z^D}
		       {1 - c^B_{D}z - c^B_{D+1}z^2 - \cdots - c^B_{2D-1}z^D}
		\f]
		となる.
*/
template <u_int D, class T> inline BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T cF[D+D], const T cB[D+D])
{
    _iirF.initialize(cF);
    _iirB.initialize(cB);
#ifdef _DEBUG
  /*T	limit0, limit1, limit2;
    limits(limit0, limit1, limit2);
    std::cerr << "limit0 = " << limit0 << ", limit1 = " << limit1
    << ", limit2 = " << limit2 << std::endl;*/
#endif
    return *this;
}

//! 両側フィルタのz変換係数をセットする
/*!
  \param c	前進方向z変換係数. z変換関数は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
  \param order	フィルタの微分階数． #Zeroth または #Second ならば対称フィルタ
		として， #First ならば反対称フィルタとして自動的に後退方向の
		z変換係数を計算する． #Zeroth, #First, #Second のときに，それ
		ぞれ in(n) = 1, in(n) = n, in(n) = n^2 に対する出力が
		1, 1, 2になるよう，全体のスケールも調整される．
  \return	このフィルタ自身
*/
template <u_int D, class T> BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T c[D+D], Order order)
{
  // Compute 0th, 1st and 2nd derivatives of the forward z-transform
  // functions at z = 1.
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (u_int i = 0; i < D; ++i)
    {
	n0 +=	      c[i];
	d0 -=	      c[D+i];
	n1 +=	    i*c[D-1-i];
	d1 -=	(i+1)*c[D+D-1-i];
	n2 += (i-1)*i*c[D-1-i];
	d2 -= i*(i+1)*c[D+D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    
  // Compute denominators.
    T	cF[D+D], cB[D+D];
    for (u_int i = 0; i < D; ++i)
	cB[D+D-1-i] = cF[D+i] = c[D+i];

  // Compute nominators.
    if (order == First)	// Antisymmetric filter
    {
	const T	k = -0.5/x1;
	cF[D-1] = cB[D-1] = 0;
	for (u_int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = -cF[i];				// i(n+D-1-i)
	}
    }
    else		// Symmetric filter
    {
	const T	k = (order == Second ? 1.0 / (x1 + x2)
				     : 1.0 / (2.0*x0 - c[D-1]));
	cF[D-1] = k*c[D-1];				// i(n)
	cB[D-1] = cF[D-1] * c[D];			// i(n+D)
	for (u_int i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = cF[i] + cF[D-1] * cF[D+1+i];	// i(n+D-1-i)
	}
    }

    return initialize(cF, cB);
}
    
//! 特定の入力データ列に対してフィルタを適用した場合の極限値を求める
/*!
  \param limit0		一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1		傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2		2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <u_int D, class T> void
BidirectionalIIRFilter<D, T>::limits(T& limit0, T& limit1, T& limit2) const
{
    T	limit0F, limit1F, limit2F;
    _iirF.limitsF(limit0F, limit1F, limit2F);

    T	limit0B, limit1B, limit2B;
    _iirB.limitsB(limit0B, limit1B, limit2B);

    limit0 = limit0F + limit0B;
    limit1 = limit1F + limit1B;
    limit2 = limit2F + limit2B;
}

//! フィルタによる畳み込みを行う. 
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
*/
template <u_int D, class T> template <class IN, class OUT> inline OUT
BidirectionalIIRFilter<D, T>::convolve(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    typedef Array<value_type, Buf<value_type, true> >		buf_type;
    typedef typename buf_type::iterator				buf_iterator;
    
    buf_type	bufF(std::distance(ib, ie)), bufB(bufF.size());

    _iirF.forward(ib, ie, bufF.begin());
    _iirB.backward(std::reverse_iterator<IN>(ie),
		   std::reverse_iterator<IN>(ib),
		   std::reverse_iterator<buf_iterator>(bufB.end()));
    return std::transform(bufF.cbegin(), bufF.cend(), bufB.cbegin(),
			  out, std::plus<value_type>());
}

//! 与えられた長さの入力データ列に対する出力データ列の長さを返す
/*!
  \param inLength	入力データ列の長さ
  \return		出力データ列の長さ
*/
template <u_int D, class T> inline u_int
BidirectionalIIRFilter<D, T>::outLength(u_int inLength)
{
    return IIRFilter<D, T>::outLength(inLength);
}
    
/************************************************************************
*  class BidirectionalIIRFilter2<D, T>					*
************************************************************************/
//! 2次元両側Infinite Inpulse Response Filterを表すクラス
template <u_int D, class T=float>
class BidirectionalIIRFilter2
    : public SeparableFilter2<BidirectionalIIRFilter<D, T> >
{
  private:
    typedef BidirectionalIIRFilter<D, T>	biir_type;
    typedef SeparableFilter2<biir_type>		super;

  public:
    typedef typename biir_type::coeff_type	coeff_type;
    typedef typename biir_type::coeffs_type	coeffs_type;
    typedef typename biir_type::Order		Order;
    
  public:
    BidirectionalIIRFilter2&
			initialize(const T cHF[], const T cHB[],
				   const T cVF[], const T cVB[])	;
    BidirectionalIIRFilter2&
			initialize(const T cHF[], Order orderH,
				   const T cVF[], Order orderV)		;

    using		super::convolve;
    using		super::filterH;
    using		super::filterV;
    
    const coeffs_type&	ciHF()		const	{return filterH().ciF();}
    const coeffs_type&	coHF()		const	{return filterH().coF();}
    const coeffs_type&	ciHB()		const	{return filterH().ciB();}
    const coeffs_type&	coHB()		const	{return filterH().coB();}
    const coeffs_type&	ciVF()		const	{return filterV().ciF();}
    const coeffs_type&	coVF()		const	{return filterV().coF();}
    const coeffs_type&	ciVB()		const	{return filterV().ciB();}
    const coeffs_type&	coVB()		const	{return filterV().coB();}
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param cHB	横方向後退z変換係数
  \param cVF	縦方向前進z変換係数
  \param cVB	縦方向後退z変換係数
  \return	このフィルタ自身
*/
template <u_int D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], const T cHB[],
					  const T cVF[], const T cVB[])
{
    filterH().initialize(cHF, cHB);
    filterV().initialize(cVF, cVB);

    return *this;
}

//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param orderH 横方向微分階数
  \param cVF	縦方向前進z変換係数
  \param orderV	縦方向微分階数
  \return	このフィルタ自身
*/
template <u_int D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], Order orderH,
					  const T cVF[], Order orderV)
{
    filterH().initialize(cHF, orderH);
    filterV().initialize(cVF, orderV);

    return *this;
}

}
#endif	/* !__TUIIRFilterPP_h */
