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
 *  $Id: DericheConvolver.h,v 1.2 2012-09-15 03:59:19 ueshiba Exp $
 */
/*!
  \file		DericheConvolver.h
  \brief	Canny-Deriche核による畳み込みに関するクラスの定義と実装
*/
#ifndef	__TUDericheConvolver_h
#define	__TUDericheConvolver_h

#include "TU/IIRFilter.h"

namespace TU
{
/************************************************************************
*  class DericheCoefficients<T>						*
************************************************************************/
//! Canny-Deriche核の係数を表すクラス
template <class T> class DericheCoefficients
{
  public:
    typedef T	value_type;
    
    void	initialize(T alpha)			;
    
  protected:
    DericheCoefficients(T alpha)			{initialize(alpha);}
    
  protected:
    value_type	_c0[4];		//!< forward coefficients for smoothing
    value_type	_c1[4];		//!< forward coefficients for 1st derivatives
    value_type	_c2[4];		//!< forward coefficients for 2nd derivatives
};

//! Canny-Deriche核の初期化を行う
/*!
  \param alpha	フィルタサイズを表す正数（小さいほど広がりが大きい）
*/
template <class T> inline void
DericheCoefficients<T>::initialize(value_type alpha)
{
    const value_type	e  = std::exp(-alpha), beta = std::sinh(alpha);
    _c0[0] =  (alpha - 1.0) * e;		// i(n-1)
    _c0[1] =  1.0;				// i(n)
    _c0[2] = -e * e;				// oF(n-2)
    _c0[3] =  2.0 * e;				// oF(n-1)

    _c1[0] = -1.0;				// i(n-1)
    _c1[1] =  0.0;				// i(n)
    _c1[2] = -e * e;				// oF(n-2)
    _c1[3] =  2.0 * e;				// oF(n-1)

    _c2[0] =  (1.0 + beta) * e;			// i(n-1)
    _c2[1] = -1.0;				// i(n)
    _c2[2] = -e * e;				// oF(n-2)
    _c2[3] =  2.0 * e;				// oF(n-1)
}

/************************************************************************
*  class DericheConvoler<T>						*
************************************************************************/
//! Canny-Deriche核による1次元配列畳み込みを行うクラス
template <class T> class DericheConvolver
    : public DericheCoefficients<T>, private BidirectionalIIRFilter<2u, T>
{
  public:
    typedef T						coeff_type;
    
  private:
    typedef DericheCoefficients<T>			coeffs;
    typedef BidirectionalIIRFilter<2u, T>		super;
    
  public:
    DericheConvolver(T alpha=1)	:DericheCoefficients<T>(alpha)		{}

    DericheConvolver&	initialize(T alpha)				;
    
    template <class IN, class OUT> OUT	smooth(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diff  (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diff2 (IN ib, IN ie, OUT out)	;

  protected:
    using	coeffs::_c0;
    using	coeffs::_c1;
    using	coeffs::_c2;
};

//! Canny-Deriche核のalpha値を設定する
/*!
  \param sigma	alpha値
  \return	このガウス核
*/
template <class T> DericheConvolver<T>&
DericheConvolver<T>::initialize(T alpha)
{
    coeffs::initialize(alpha);
    return *this;
}
    
//! Canny-Deriche核によるスムーシング
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver<T>::smooth(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, super::Zeroth)(ib, ie, out);
}

//! Canny-Deriche核による1階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver<T>::diff(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, super::First)(ib, ie, out);
}

//! Canny-Deriche核による2階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver<T>::diff2(IN ib, IN ie, OUT out)
{
    return super::initialize(_c2, super::Second)(ib, ie, out);
}

/************************************************************************
*  class DericheConvoler2<T>						*
************************************************************************/
//! Canny-Deriche核による2次元配列畳み込みを行うクラス
template <class T> class DericheConvolver2
    : public DericheCoefficients<T>, private BidirectionalIIRFilter2<2u, T>
{
  public:
    typedef T						coeff_type;
    
  private:
    typedef DericheCoefficients<T>			coeffs;
    typedef BidirectionalIIRFilter2<2u, T>		super;
    typedef BidirectionalIIRFilter<2u, T>		IIRF;
    
  public:
    DericheConvolver2(T alpha=1)	:DericheCoefficients<T>(alpha)	{}

    DericheConvolver2&	initialize(T alpha)				;
    
    template <class IN, class OUT> OUT	smooth(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diffH (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diffV (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diffHH(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diffHV(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diffVV(IN ib, IN ie, OUT out)	;

  protected:
    using	coeffs::_c0;
    using	coeffs::_c1;
    using	coeffs::_c2;
};

//! Canny-Deriche核のalpha値を設定する
/*!
  \param sigma	alpha値
  \return	このガウス核
*/
template <class T> DericheConvolver2<T>&
DericheConvolver2<T>::initialize(T alpha)
{
    coeffs::initialize(alpha);
    return *this;
}
    
//! Canny-Deriche核によるスムーシング
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::smooth(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Canny-Deriche核による横方向1階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::diffH(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, IIRF::First,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Canny-Deriche核による縦方向1階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::diffV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c1, IIRF::First)(ib, ie, out);
}

//! Canny-Deriche核による横方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::diffHH(IN ib, IN ie, OUT out)
{
    return super::initialize(_c2, IIRF::Second,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Canny-Deriche核による縦横両方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::diffHV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, IIRF::First,
			     _c1, IIRF::First)(ib, ie, out);
}

//! Canny-Deriche核による縦方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
DericheConvolver2<T>::diffVV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c2, IIRF::Second)(ib, ie, out);
}

}
#endif	/* !__TUDericheConvolver_h */
