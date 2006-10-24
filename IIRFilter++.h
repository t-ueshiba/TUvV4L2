/*
 *  $Id: IIRFilter++.h,v 1.3 2006-10-24 08:10:50 ueshiba Exp $
 */
#ifndef __TUIIRFilterPP_h
#define __TUIIRFilterPP_h

#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class IIRFilter							*
************************************************************************/
/*!
  片側Infinite Inpulse Response Filterを表すクラス．
*/
template <u_int D> class IIRFilter
{
  public:
    IIRFilter&	initialize(const float c[D+D])				;
    template <class S> const IIRFilter&
		forward(const Array<S>& in, Array<float>& out)	const	;
    template <class S> const IIRFilter&
		backward(const Array<S>& in, Array<float>& out)	const	;
    void	limitsF(float& limit0F,
			float& limit1F, float& limit2F)		const	;
    void	limitsB(float& limit0B,
			float& limit1B, float& limit2B)		const	;
    
  private:
    float	_c[D+D];	// coefficients
};

/************************************************************************
*  class BilateralIIRFilter						*
************************************************************************/
/*!
  両側Infinite Inpulse Response Filterを表すクラス．
*/
template <u_int D> class BilateralIIRFilter
{
  public:
  //! 微分の階数
    enum Order
    {
	Zeroth,						//!< 0階微分
	First,						//!< 1階微分
	Second						//!< 2階微分
    };
    
    BilateralIIRFilter&	initialize(const float cF[D+D], const float cB[D+D]);
    BilateralIIRFilter&	initialize(const float c[D+D], Order order)	;
    template <class S>
    BilateralIIRFilter&	convolve(const Array<S>& in)			;
    u_int		dim()					const	;
    float		operator [](int i)			const	;
    void		limits(float& limit0,
			       float& limit1,
			       float& limit2)			const	;
    
  private:
    IIRFilter<D>	_iirF;
    Array<float>	_bufF;
    IIRFilter<D>	_iirB;
    Array<float>	_bufB;
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
		  H^B(z) = \frac{c^B_{0}z + c^B_{1}z^2 + \cdots + c^B_{D-1}*z^D}
		       {1 - c^B_{D}z - c^B_{D+1}z^2 - \cdots - c^B_{2D-1}*z^D}
		\f]
		となる.
*/
template <u_int D> inline BilateralIIRFilter<D>&
BilateralIIRFilter<D>::initialize(const float cF[D+D], const float cB[D+D])
{
    _iirF.initialize(cF);
    _iirB.initialize(cB);
#ifdef DEBUG
    float	limit0, limit1, limit2;
    limits(limit0, limit1, limit2);
    std::cerr << "limit0 = " << limit0 << ", limit1 = " << limit1
	      << ", limit2 = " << limit2 << std::endl;
#endif
    return *this;
}

//! フィルタによる畳み込みを行う. 出力は operator [](int) で取り出す
/*!
  \param in	入力データ列.
  return	このフィルタ自身.
*/
template <u_int D> template <class S> inline BilateralIIRFilter<D>&
BilateralIIRFilter<D>::convolve(const Array<S>& in)
{
    _iirF.forward(in, _bufF);
    _iirB.backward(in, _bufB);

    return *this;
}

//! 畳み込みの出力データ列の次元を返す
/*!
  \return	出力データ列の次元.
*/
template <u_int D> inline u_int
BilateralIIRFilter<D>::dim() const
{
    return _bufF.dim();
}

//! 畳み込みの出力データの特定の要素を返す
/*!
  \param i	要素のindex.
  \return	要素の値.
*/
template <u_int D> inline float
BilateralIIRFilter<D>::operator [](int i) const
{
    return _bufF[i] + _bufB[i];
}

/************************************************************************
*  class BilateralIIRFilter2						*
************************************************************************/
/*!
  2次元両側Infinite Inpulse Response Filterを表すクラス．
*/
template <u_int D> class BilateralIIRFilter2
{
  public:
    typedef typename BilateralIIRFilter<D>::Order	Order;
    
    BilateralIIRFilter2&
		initialize(float cHF[D+D], float cHB[D+D],
			   float cVF[D+D], float cVB[D+D])		;
    BilateralIIRFilter2&
		initialize(float cHF[D+D], Order orderH,
			   float cVF[D+D], Order orderV)		;
    template <class S, class T> BilateralIIRFilter2&
		convolve(const Image<S>& in, Image<T>& out)		;
    
  private:
    BilateralIIRFilter<D>	_iirH;
    BilateralIIRFilter<D>	_iirV;
    Array2<Array<float> >	_buf;
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数.
  \param cHB	横方向後退z変換係数.
  \param cHV	縦方向前進z変換係数.
  \param cHV	縦方向後退z変換係数.
  \return	このフィルタ自身.
*/
template <u_int D> inline BilateralIIRFilter2<D>&
BilateralIIRFilter2<D>::initialize(float cHF[D+D], float cHB[D+D],
				   float cVF[D+D], float cVB[D+D])
{
    _iirH.initialize(cHF, cHB);
    _iirV.initialize(cVF, cVB);

    return *this;
}

//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数.
  \param orderH 横方向微分階数.
  \param cHV	縦方向前進z変換係数.
  \param orderV	縦方向微分階数.
  \return	このフィルタ自身.
*/
template <u_int D> inline BilateralIIRFilter2<D>&
BilateralIIRFilter2<D>::initialize(float cHF[D+D], Order orderH,
				   float cVF[D+D], Order orderV)
{
    _iirH.initialize(cHF, orderH);
    _iirV.initialize(cVF, orderV);

    return *this;
}

/************************************************************************
*  class DericheConvoler						*
************************************************************************/
/*!
  Canny-Deriche核によるスムーシング，1次微分および2次微分を含む
  画像畳み込みを行うクラス．
*/
class DericheConvolver : private BilateralIIRFilter2<2u>
{
  public:
    using	BilateralIIRFilter2<2u>::Order;
    
    DericheConvolver(float alpha=1.0)		{initialize(alpha);}

    DericheConvolver&	initialize(float alpha)				;
    template <class S, class T>
    DericheConvolver&	smooth(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    DericheConvolver&	gradH(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    DericheConvolver&	gradV(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    DericheConvolver&	laplacian(const Image<S>& in, Image<T>& out)	;

  private:
    float		_c0[4];	// forward coefficients for smoothing
    float		_c1[4];	// forward coefficients for 1st derivatives
    float		_c2[4];	// forward coefficients for 2nd derivatives
    Image<float>	_tmp;	// buffer for storing intermediate values
};

//! Canny-Deriche核によるスムーシング
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このCanny-Deriche核自身.
*/
template <class S, class T> inline DericheConvolver&
DericheConvolver::smooth(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による横方向1次微分
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このCanny-Deriche核自身.
*/
template <class S, class T> inline DericheConvolver&
DericheConvolver::gradH(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c1, BilateralIIRFilter<2u>::First,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, out);

    return *this;
}

//! Canny-Deriche核による縦方向1次微分
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このCanny-Deriche核自身.
*/
template <class S, class T> inline DericheConvolver&
DericheConvolver::gradV(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c1, BilateralIIRFilter<2u>::First).convolve(in, out);

    return *this;
}

//! Canny-Deriche核によるラプラシアン
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このCanny-Deriche核自身.
*/
template <class S, class T> inline DericheConvolver&
DericheConvolver::laplacian(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<2u>::
	initialize(_c2, BilateralIIRFilter<2u>::Second,
		   _c0, BilateralIIRFilter<2u>::Zeroth).convolve(in, _tmp).
	initialize(_c0, BilateralIIRFilter<2u>::Zeroth,
		   _c2, BilateralIIRFilter<2u>::Second).convolve(in, out);
    out += _tmp;
    
    return *this;
}

/************************************************************************
*  class GaussianConvoler						*
************************************************************************/
/*!
  Gauss核によるスムーシング，1次微分(DOG)および2次微分(LOG)を含む
  画像畳み込みを行うクラス．
*/
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
	typedef double		T;
	typedef Array<Params>	AT;

	EvenConstraint(T sigma) :_sigma(sigma)				{}
	
	Vector<T>	operator ()(const AT& params)		const	;
	Matrix<T>	jacobian(const AT& params)		const	;

      private:
	T		_sigma;
    };

    class CostFunction
    {
      public:
	typedef double		T;
	typedef Array<Params>	AT;
    
	enum			{D = 2};

	CostFunction(int ndivisions, T range)
	    :_ndivisions(ndivisions), _range(range)			{}
    
	Vector<T>	operator ()(const AT& params)		const	;
	Matrix<T>	jacobian(const AT& params)		const	;
	void		update(AT& params, const Vector<T>& dp)	const	;

      private:
	const int	_ndivisions;
	const T		_range;
    };

  public:
    GaussianConvolver(float sigma=1.0)		{initialize(sigma);}

    GaussianConvolver&	initialize(float sigma)				;
    template <class S, class T>
    GaussianConvolver&	smooth(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    GaussianConvolver&	gradH(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    GaussianConvolver&	gradV(const Image<S>& in, Image<T>& out)	;
    template <class S, class T>
    GaussianConvolver&	laplacian(const Image<S>& in, Image<T>& out)	;

  private:
    float		_c0[8];	// forward coefficients for smoothing
    float		_c1[8];	// forward coefficients for 1st derivatives
    float		_c2[8];	// forward coefficients for 2nd derivatives
    Image<float>	_tmp;	// buffer for storing intermediate values
};

//! Gauss核によるスムーシング
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このGauss核自身.
*/
template <class S, class T> inline GaussianConvolver&
GaussianConvolver::smooth(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, out);

    return *this;
}

//! Gauss核による横方向1次微分(DOG)
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このGauss核自身.
*/
template <class S, class T> inline GaussianConvolver&
GaussianConvolver::gradH(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c1, BilateralIIRFilter<4u>::First,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, out);

    return *this;
}

//! Gauss核による縦方向1次微分(DOG)
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このGauss核自身.
*/
template <class S, class T> inline GaussianConvolver&
GaussianConvolver::gradV(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c1, BilateralIIRFilter<4u>::First).convolve(in, out);

    return *this;
}

//! Gauss核によるラプラシアン(LOG)
/*!
  \param in	入力画像.
  \param out	出力画像.
  \return	このGauss核自身.
*/
template <class S, class T> inline GaussianConvolver&
GaussianConvolver::laplacian(const Image<S>& in, Image<T>& out)
{
    BilateralIIRFilter2<4u>::
	initialize(_c2, BilateralIIRFilter<4u>::Second,
		   _c0, BilateralIIRFilter<4u>::Zeroth).convolve(in, _tmp).
	initialize(_c0, BilateralIIRFilter<4u>::Zeroth,
		   _c2, BilateralIIRFilter<4u>::Second).convolve(in, out);
    out += _tmp;
    
    return *this;
}

}

#endif	// !__TUIIRFilterPP_h
