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
  \file		GaussianConvolver.h
  \brief	Gauss核による畳み込みに関するクラスの定義と実装
*/
#ifndef	__TU_GAUSSIANCONVOLVER_H
#define	__TU_GAUSSIANCONVOLVER_H

#include "TU/Vector++.h"
#include "TU/IIRFilter.h"

namespace TU
{
/************************************************************************
*  class GaussianCoefficients<T>					*
************************************************************************/
//! Gauss核の係数を表すクラス
template <class T> class GaussianCoefficients
{
  private:
    typedef double			element_type;
    typedef Matrix<element_type>	matrix_type;
    typedef Vector<element_type>	vector_type;

    struct Params
    {
	void		set(element_type aa, element_type bb,
			    element_type tt, element_type aaa);
	Params&		operator -=(const vector_type& p)		;
    
	element_type	a, b, theta, alpha;
    };

    class EvenConstraint
    {
      public:
	typedef Array<Params>	argument_type;

	EvenConstraint(element_type sigma) :_sigma(sigma)		{}
	
	vector_type	operator ()(const argument_type& params) const	;
	matrix_type	jacobian(const argument_type& params)	 const	;

      private:
	element_type	_sigma;
    };

    class CostFunction
    {
      public:
	typedef typename GaussianCoefficients<T>::element_type	element_type;
	typedef Array<Params>					argument_type;
    
	enum		{D = 2};

	CostFunction(int ndivisions, element_type range)
	    :_ndivisions(ndivisions), _range(range)			{}
    
	vector_type	operator ()(const argument_type& params) const	;
	matrix_type	jacobian(const argument_type& params)	 const	;
	void		update(argument_type& params,
			       const vector_type& dp)		 const	;

      private:
	const int		_ndivisions;
	const element_type	_range;
    };

  public:
    typedef	T					coeff_type;
    
  public:
    void	initialize(coeff_type sigma)		;
    
  protected:
    GaussianCoefficients(coeff_type sigma)		{initialize(sigma);}
    
  protected:
    coeff_type	_c0[8];		//!< forward coefficients for smoothing
    coeff_type	_c1[8];		//!< forward coefficients for 1st derivatives
    coeff_type	_c2[8];		//!< forward coefficients for 2nd derivatives
};
    
/************************************************************************
*  class GaussianConvolver<T>						*
************************************************************************/
//! Gauss核による1次元配列畳み込みを行うクラス
template <class T> class GaussianConvolver
    : public GaussianCoefficients<T>, private BidirectionalIIRFilter<4u, T>
{
  public:
    typedef T						coeff_type;
    
  private:
    typedef GaussianCoefficients<coeff_type>		coeffs;
    typedef BidirectionalIIRFilter<4u, coeff_type>	super;
    
  public:
    GaussianConvolver(coeff_type sigma=1.0)	:coeffs(sigma)		{}

    GaussianConvolver&	initialize(coeff_type sigma)			;

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
template <class T> GaussianConvolver<T>&
GaussianConvolver<T>::initialize(coeff_type sigma)
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
template <class T> template <class IN, class OUT> inline void
GaussianConvolver<T>::smooth(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, super::Zeroth).convolve(ib, ie, out);
}

//! Gauss核による1階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver<T>::diff(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1, super::First).convolve(ib, ie, out);
}

//! Gauss核による2階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver<T>::diff2(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c2, super::Second).convolve(ib, ie, out);
}

/************************************************************************
*  class GaussianConvolver2<T>						*
************************************************************************/
//! Gauss核による2次元配列畳み込みを行うクラス
template <class T> class GaussianConvolver2
    : public GaussianCoefficients<T>, private BidirectionalIIRFilter2<4u, T>
{
  public:
    typedef T						coeff_type;
    
  private:
    typedef GaussianCoefficients<coeff_type>		coeffs;
    typedef BidirectionalIIRFilter2<4u, coeff_type>	super;
    typedef BidirectionalIIRFilter<4u, coeff_type>	IIRF;
    
  public:
    GaussianConvolver2(coeff_type sigma=1.0)	:coeffs(sigma)		{}

    GaussianConvolver2&	initialize(coeff_type sigma)			;
    using		super::grainSize;
    using		super::setGrainSize;
    
    template <class IN, class OUT>
    void		smooth(IN ib, IN ie, OUT out)			;
    template <class IN, class OUT>
    void		diffH (IN ib, IN ie, OUT out)			;
    template <class IN, class OUT>
    void		diffV (IN ib, IN ie, OUT out)			;
    template <class IN, class OUT>
    void		diffHH(IN ib, IN ie, OUT out)			;
    template <class IN, class OUT>
    void		diffHV(IN ib, IN ie, OUT out)			;
    template <class IN, class OUT>
    void		diffVV(IN ib, IN ie, OUT out)			;
};

//! Gauss核のsigma値を設定する
/*!
  \param sigma	sigma値
  \return	このガウス核
*/
template <class T> GaussianConvolver2<T>&
GaussianConvolver2<T>::initialize(coeff_type sigma)
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
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::smooth(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, IIRF::Zeroth, coeffs::_c0, IIRF::Zeroth)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による横方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::diffH(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1, IIRF::First, coeffs::_c0, IIRF::Zeroth)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::diffV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, IIRF::Zeroth, coeffs::_c1, IIRF::First)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による横方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::diffHH(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c2, IIRF::Second, coeffs::_c0, IIRF::Zeroth)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦横両方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::diffHV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c1, IIRF::First, coeffs::_c1, IIRF::First)
	  .template convolve<IN, OUT>(ib, ie, out);
}

//! Gauss核による縦方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
GaussianConvolver2<T>::diffVV(IN ib, IN ie, OUT out)
{
    super::initialize(coeffs::_c0, IIRF::Zeroth, coeffs::_c2, IIRF::Second)
	  .template convolve<IN, OUT>(ib, ie, out);
}

}
#endif	// !__TU_GAUSSIANCONVOLVER_H
