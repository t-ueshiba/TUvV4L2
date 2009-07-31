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
 *  $Id: DericheConvolver.h,v 1.7 2009-07-31 07:04:44 ueshiba Exp $
 */
#ifndef	__TUDericheConvolver_h
#define	__TUDericheConvolver_h

#include "TU/IIRFilter.h"

namespace TU
{
/************************************************************************
*  class DericheCoefficients						*
************************************************************************/
class DericheCoefficients
{
  public:
    void	initialize(float alpha)			;
    
  protected:
    DericheCoefficients(float alpha)			{initialize(alpha);}
    
  protected:
    float	_c0[4];		// forward coefficients for smoothing
    float	_c1[4];		// forward coefficients for 1st derivatives
    float	_c2[4];		// forward coefficients for 2nd derivatives
};

//! Canny-Deriche核の初期化を行う
/*!
  \param alpha	フィルタサイズを表す正数（小さいほど広がりが大きい）
*/
inline void
DericheCoefficients::initialize(float alpha)
{
    const float	e  = expf(-alpha), beta = sinhf(alpha);
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
*  class DericheConvoler						*
************************************************************************/
//! Canny-Deriche核による1次元配列畳み込みを行うクラス
class DericheConvolver
    : public DericheCoefficients, private BilateralIIRFilter<2u>
{
  public:
    typedef BilateralIIRFilter<2u>		BIIRF;
    
    DericheConvolver(float alpha=1.0)	:DericheCoefficients(alpha)	{}

    template <class T1, class B1, class T2, class B2> DericheConvolver&
	smooth(const Array<T1, B1>& in, Array<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diff(const Array<T1, B1>& in, Array<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> DericheConvolver&
	diff2(const Array<T1, B1>& in, Array<T2, B2>& out)		;
};

//! Canny-Deriche核によるスムーシング
/*!
  \param in	入力1次元配列
  \param out	出力1次元配列
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2>
inline DericheConvolver&
DericheConvolver::smooth(const Array<T1, B1>& in, Array<T2, B2>& out)
{
    BIIRF::initialize(_c0, BIIRF::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による1階微分
/*!
  \param in	入力1次元配列
  \param out	出力1次元配列
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2>
inline DericheConvolver&
DericheConvolver::diff(const Array<T1, B1>& in, Array<T2, B2>& out)
{
    BIIRF::initialize(_c1, BIIRF::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による2階微分
/*!
  \param in	入力1次元配列
  \param out	出力1次元配列
  \return	このCanny-Deriche核自身
*/
template <class T1, class B1, class T2, class B2>
inline DericheConvolver&
DericheConvolver::diff2(const Array<T1, B1>& in, Array<T2, B2>& out)
{
    BIIRF::initialize(_c2, BIIRF::Second).convolve(in, out);

    return *this;
}

/************************************************************************
*  class DericheConvoler2<BIIRH, BIIRV>					*
************************************************************************/
//! Canny-Deriche核による2次元配列畳み込みを行うクラス
template <class BIIRH=BilateralIIRFilter<2u>, class BIIRV=BIIRH>
class DericheConvolver2
    : public DericheCoefficients, private BilateralIIRFilter2<BIIRH, BIIRV>
{
  public:
    typedef BilateralIIRFilter<2u>		BIIRF;
    typedef BilateralIIRFilter2<BIIRH, BIIRV>	BIIRF2;
    
    DericheConvolver2(float alpha=1.0)	:DericheCoefficients(alpha)	{}
    DericheConvolver2(float alpha, u_int nthreads)
    	:DericheCoefficients(alpha), BIIRF2(nthreads)			{}

    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	smooth(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	diffH(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	diffV(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	diffHH(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	diffHV(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	diffVV(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out)	;
    template <class T1, class B1, class R1, class T2, class B2, class R2>
    DericheConvolver2&
	laplacian(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out);

  private:
    Array2<Array<float> >	_tmp;	// buffer for computing Laplacian
};

//! Canny-Deriche核によるスムーシング
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::smooth(const Array2<T1, B1, R1>& in,
					Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c0, BIIRF::Zeroth,
		       _c0, BIIRF::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による横方向1階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::diffH(const Array2<T1, B1, R1>& in,
				       Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c1, BIIRF::First,
		       _c0, BIIRF::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦方向1階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::diffV(const Array2<T1, B1, R1>& in,
				       Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c0, BIIRF::Zeroth,
		       _c1, BIIRF::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による横方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::diffHH(const Array2<T1, B1, R1>& in,
					Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c2, BIIRF::Second,
		       _c0, BIIRF::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦横両方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::diffHV(const Array2<T1, B1, R1>& in,
					Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c1, BIIRF::First,
		       _c1, BIIRF::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::diffVV(const Array2<T1, B1, R1>& in,
					Array2<T2, B2, R2>& out)
{
    BIIRF2::initialize(_c0, BIIRF::Zeroth,
		       _c2, BIIRF::Second).convolve(in, out);

    return *this;
}

//! Canny-Deriche核によるラプラシアン
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このCanny-Deriche核自身
*/
template <class BIIRH, class BIIRV>
template <class T1, class B1, class R1, class T2, class B2, class R2>
inline DericheConvolver2<BIIRH, BIIRV>&
DericheConvolver2<BIIRH, BIIRV>::laplacian(const Array2<T1, B1, R1>& in,
					   Array2<T2, B2, R2>& out)
{
    diffHH(in, _tmp).diffVV(in, out);
    out += _tmp;
    
    return *this;
}

}

#endif	/* !__TUDericheConvolver_h */
