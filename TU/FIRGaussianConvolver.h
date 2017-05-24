/*!
  \file		FIRGaussianConvolver.h
  \author	Toshio UESHIBA
  \brief	Gauss核による畳み込みに関連するクラスの定義と実装
*/ 
#ifndef	TU_FIRGAUSSIANCONVOLVER_H
#define	TU_FIRGAUSSIANCONVOLVER_H

#include "TU/FIRFilter.h"

namespace TU
{
/************************************************************************
*  class FIRGaussianCoefficients<D, T>					*
************************************************************************/
//! Gauss核の係数を表すクラス
template <size_t D, class T>
class FIRGaussianCoefficients
{
  public:
    typedef	T					coeff_type;
    
  public:
    void	initialize(coeff_type sigma)		;
    
  protected:
    FIRGaussianCoefficients(coeff_type sigma)		{initialize(sigma);}
    
  protected:
    coeff_type	_c0[2*D+1];		//!< coefficients for smoothing
    coeff_type	_c1[2*D+1];		//!< coefficients for 1st derivatives
    coeff_type	_c2[2*D+1];		//!< coefficients for 2nd derivatives
};
    
/************************************************************************
*  class FIRGaussianConvolver<D, T>					*
************************************************************************/
//! Gauss核による1次元配列畳み込みを行うクラス
template <size_t D=16, class T=float>
class FIRGaussianConvolver : public  FIRGaussianCoefficients<D, T>,
			     private FIRFilter<2*D+1, T>
{
  private:
    typedef FIRGaussianCoefficients<D, T>	coeffs;
    typedef FIRFilter<2*D+1, T>			super;
    
  public:
    typedef T					coeff_type;
    
  public:
    FIRGaussianConvolver(coeff_type sigma=1.0)	:coeffs(sigma)		{}

    FIRGaussianConvolver&	initialize(coeff_type sigma)		;
    using			super::winSize;
    using			super::outLength;

    template <class IN, class OUT> void	smooth(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diff  (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diff2 (IN ib, IN ie, OUT out)	;

  protected:
    using	coeffs::_c0;
    using	coeffs::_c1;
    using	coeffs::_c2;
};

//! Gauss核のsigma値を設定する
/*!
  \param sigma	sigma値
  \return	このガウス核
*/
template <size_t D, class T> FIRGaussianConvolver<D, T>&
FIRGaussianConvolver<D, T>::initialize(coeff_type sigma)
{
    coeffs::initialize(sigma);
    return *this;
}
    
//! Gauss核によるスムーシング
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver<D, T>::smooth(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0).convolve(ib, ie, out);
}

//! Gauss核による1階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver<D, T>::diff(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1).convolve(ib, ie, out);
}

//! Gauss核による2階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver<D, T>::diff2(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c2).convolve(ib, ie, out);
}

/************************************************************************
*  class FIRGaussianConvolver2<D, T>					*
************************************************************************/
//! Gauss核による2次元配列畳み込みを行うクラス
template <size_t D=16, class T=float>
class FIRGaussianConvolver2 : public  FIRGaussianCoefficients<D, T>,
			      private FIRFilter2<2*D+1, T>
{
  private:
    typedef FIRGaussianCoefficients<D, T>	coeffs;
    typedef FIRFilter2<2*D+1, T>		super;
    
  public:
    typedef T					coeff_type;
    
  public:
    FIRGaussianConvolver2(coeff_type sigma=1.0)	:coeffs(sigma)		{}

    FIRGaussianConvolver2&	initialize(coeff_type sigma)		;
    using			super::grainSize;
    using			super::setGrainSize;
    using			super::winSize;
    using			super::outLength;
    
    template <class IN, class OUT> void	smooth(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffH (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffV (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffHH(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffHV(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffVV(IN ib, IN ie, OUT out)	;
};

//! Gauss核のsigma値を設定する
/*!
  \param sigma	sigma値
  \return	このガウス核
*/
template <size_t D, class T> FIRGaussianConvolver2<D, T>&
FIRGaussianConvolver2<D, T>::initialize(coeff_type sigma)
{
    coeffs::initialize(sigma);
    return *this;
}
    
//! Gauss核によるスムーシング
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::smooth(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, coeffs::_c0)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による横方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::diffH(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1, coeffs::_c0)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::diffV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, coeffs::_c1)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による横方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::diffHH(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c2, coeffs::_c0)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦横両方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::diffHV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1, coeffs::_c1)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<D, T>::diffVV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, coeffs::_c2)
	  .template convolve<IN, OUT>(ib, ie, out);
}
    
}
#endif	// !TU_FIRGAUSSIANCONVOLVER_H
