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
 *  $Id: GaussianConvolver.h,v 1.1 2008-08-07 07:26:48 ueshiba Exp $
 */
#ifndef	__TUGaussianConvolver_h
#define	__TUGaussianConvolver_h

#include "TU/Vector++.h"
#include "TU/IIRFilter++.h"

namespace TU
{
/************************************************************************
*  class GaussianConvoler						*
************************************************************************/
//! Gauss核による画像畳み込みを行うクラス
class GaussianConvolver : private BilateralIIRFilter2<4u>
{
  private:
    struct Params
    {
	void		set(double aa, double bb, double tt, double aaa);
	Params&		operator -=(const Vector<double>& p)		;
    
	double		a, b, theta, alpha;
    };

    class EvenConstraint
    {
      public:
	typedef double		ET;
	typedef Array<Params>	AT;

	EvenConstraint(ET sigma) :_sigma(sigma)				{}
	
	Vector<ET>	operator ()(const AT& params)		const	;
	Matrix<ET>	jacobian(const AT& params)		const	;

      private:
	ET		_sigma;
    };

    class CostFunction
    {
      public:
	typedef double		ET;
	typedef Array<Params>	AT;
    
	enum			{D = 2};

	CostFunction(int ndivisions, ET range)
	    :_ndivisions(ndivisions), _range(range)			{}
    
	Vector<ET>	operator ()(const AT& params)		 const	;
	Matrix<ET>	jacobian(const AT& params)		 const	;
	void		update(AT& params, const Vector<ET>& dp) const	;

      private:
	const int	_ndivisions;
	const ET	_range;
    };

  public:
    GaussianConvolver(float sigma=1.0)		{initialize(sigma);}

    GaussianConvolver&	initialize(float sigma)				;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	smooth(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	diffH(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	diffV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	diffHH(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	diffHV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	diffVV(const Array2<T1, B1>& in, Array2<T2, B2>& out)		;
    template <class T1, class B1, class T2, class B2> GaussianConvolver&
	laplacian(const Array2<T1, B1>& in, Array2<T2, B2>& out)	;

  private:
    float		_c0[8];	// forward coefficients for smoothing
    float		_c1[8];	// forward coefficients for 1st derivatives
    float		_c2[8];	// forward coefficients for 2nd derivatives
    Array2<Array<float> >
			_tmp;	// buffer for storing intermediate values
};

//! Gauss核によるスムーシング
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::smooth(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, out);

    return *this;
}

//! Gauss核による横方向1階微分(DOG)
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::diffH(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c1, BilateralIIRFilter<4u>::First,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, out);

    return *this;
}

//! Gauss核による縦方向1階微分(DOG)
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::diffV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c1, BilateralIIRFilter<4u>::First).convolve(in, out);

    return *this;
}

//! Gauss核による横方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::diffHH(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c2, BilateralIIRFilter<4u>::Second,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, out);

    return *this;
}

//! Gauss核による縦横両方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::diffHV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c1, BilateralIIRFilter<4u>::First,
		   _c1, BilateralIIRFilter<4u>::First).convolve(in, out);

    return *this;
}

//! Gauss核による縦方向2階微分
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::diffVV(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c2, BilateralIIRFilter<4u>::Second).convolve(in, out);

    return *this;
}

//! Gauss核によるラプラシアン(LOG)
/*!
  \param in	入力画像
  \param out	出力画像
  \return	このGauss核自身
*/
template <class T1, class B1, class T2, class B2> inline GaussianConvolver&
GaussianConvolver::laplacian(const Array2<T1, B1>& in, Array2<T2, B2>& out)
{
    diffHH(in, _tmp).diffVV(in, out);
    out += _tmp;
    
    return *this;
}

}

#endif	/* !__TUGaussianConvolver_h */
