/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: DericheConvolver.h,v 1.1 2008-08-07 07:26:47 ueshiba Exp $
 */
#ifndef	__TUDericheConvolver_h
#define	__TUDericheConvolver_h

#include "TU/IIRFilter++.h"

namespace TU
{
/************************************************************************
*  class DericheConvoler						*
************************************************************************/
//! Canny-Deriche核による画像畳み込みを行うクラス
class DericheConvolver : private BilateralIIRFilter2<2u>
{
  public:
    using	BilateralIIRFilter2<2u>::Order;
    
    DericheConvolver(float alpha=1.0)		{initialize(alpha);}

    DericheConvolver&	initialize(float alpha)				;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	smooth(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diffH(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diffV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diffHH(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diffHV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diffVV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	laplacian(const Array2<T1, B1>& in, Array2<T2, B2>& out)	;

  private:
    float		_c0[4];	// forward coefficients for smoothing
    float		_c1[4];	// forward coefficients for 1st derivatives
    float		_c2[4];	// forward coefficients for 2nd derivatives
    Array2<Array<float> >
			_tmp;	// buffer for storing intermediate values
};

//! Canny-Deriche核によるスムーシング
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::smooth(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による横方向1階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::diffH(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c1, BilateralIIRFilter<2u>::First,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦方向1階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::diffV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c1, BilateralIIRFilter<2u>::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による横方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::diffHH(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c2, BilateralIIRFilter<2u>::Second,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦横両方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::diffHV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c1, BilateralIIRFilter<2u>::First,
		   _c1, BilateralIIRFilter<2u>::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::diffVV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c2, BilateralIIRFilter<2u>::Second).convolve(in, out);

    return *this;
}

//! Canny-Deriche核によるラプラシアン
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2> inline DericheConvolver&
DericheConvolver::laplacian(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    diffHH(in, _tmp).diffVV(in, out);
    out += _tmp;
    
    return *this;
}

}

#endif	/* !__TUDericheConvolver_h */
