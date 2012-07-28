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
 *  $Id: GaussianConvolver.h,v 1.13 2012-07-28 09:10:11 ueshiba Exp $
 */
/*!
  \file		GaussianConvolver.h
  \brief	Gauss核による畳み込みに関するクラスの定義と実装
*/
#ifndef	__TUGaussianConvolver_h
#define	__TUGaussianConvolver_h

#include "TU/Vector++.h"
#include "TU/IIRFilter.h"

namespace TU
{
/************************************************************************
*  class GaussianCoefficients<T>					*
************************************************************************/
//! Gauss核の係数を表すクラス
template <class T> class __PORT GaussianCoefficients
{
  private:
    typedef double		value_type;
    typedef Matrix<value_type>	matrix_type;
    typedef Vector<value_type>	vector_type;

    struct Params
    {
	void		set(value_type aa, value_type bb,
			    value_type tt, value_type aaa);
	Params&		operator -=(const vector_type& p)		;
    
	value_type	a, b, theta, alpha;
    };

    class EvenConstraint
    {
      public:
	typedef Array<Params>	AT;

	EvenConstraint(value_type sigma) :_sigma(sigma)			{}
	
	vector_type	operator ()(const AT& params)		const	;
	matrix_type	jacobian(const AT& params)		const	;

      private:
	value_type	_sigma;
    };

    class CostFunction
    {
      public:
	typedef typename GaussianCoefficients<T>::value_type	value_type;
	typedef Array<Params>					AT;
    
	enum		{D = 2};

	CostFunction(int ndivisions, value_type range)
	    :_ndivisions(ndivisions), _range(range)			{}
    
	vector_type	operator ()(const AT& params)		  const	;
	matrix_type	jacobian(const AT& params)		  const	;
	void		update(AT& params, const vector_type& dp) const	;

      private:
	const int		_ndivisions;
	const value_type	_range;
    };

  public:
    void	initialize(T sigma)			;
    
  protected:
    GaussianCoefficients(T sigma)			{initialize(sigma);}
    
  protected:
    T		_c0[8];		//!< forward coefficients for smoothing
    T		_c1[8];		//!< forward coefficients for 1st derivatives
    T		_c2[8];		//!< forward coefficients for 2nd derivatives
};
    
/************************************************************************
*  class GaussianConvoler<T>						*
************************************************************************/
//! Gauss核による1次元配列畳み込みを行うクラス
template <class T> class GaussianConvolver
    : public GaussianCoefficients<T>, private BidirectionalIIRFilter<4u, T>
{
  private:
    typedef GaussianCoefficients<T>			coeffs;
    typedef BidirectionalIIRFilter<4u, T>		super;
    
  public:
    GaussianConvolver(T sigma=1.0)	:GaussianCoefficients<T>(sigma)	{}

    template <class IN, class OUT> OUT	smooth(IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diff  (IN ib, IN ie, OUT out)	;
    template <class IN, class OUT> OUT	diff2 (IN ib, IN ie, OUT out)	;

  protected:
    using	coeffs::_c0;
    using	coeffs::_c1;
    using	coeffs::_c2;
};

//! Gauss核によるスムーシング
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver<T>::smooth(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, super::Zeroth)(ib, ie, out);
}

//! Gauss核による1階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver<T>::diff(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, super::First)(ib, ie, out);
}

//! Gauss核による2階微分
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver<T>::diff2(IN ib, IN ie, OUT out)
{
    return super::initialize(_c2, super::Second)(ib, ie, out);
}

/************************************************************************
*  class GaussianConvoler2<T>						*
************************************************************************/
//! Gauss核による2次元配列畳み込みを行うクラス
template <class T> class GaussianConvolver2
    : public GaussianCoefficients<T>, private BidirectionalIIRFilter2<4u, T>
{
  private:
    typedef GaussianCoefficients<T>			coeffs;
    typedef BidirectionalIIRFilter2<4u, T>		super;
    typedef BidirectionalIIRFilter<4u, T>		IIRF;
    
  public:
    GaussianConvolver2(T sigma=1.0)	:GaussianCoefficients<T>(sigma) {}

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

//! Gauss核によるスムーシング
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::smooth(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Gauss核による横方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::diffH(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, IIRF::First,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Gauss核による縦方向1階微分(DOG)
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::diffV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c1, IIRF::First)(ib, ie, out);
}

//! Gauss核による横方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::diffHH(IN ib, IN ie, OUT out)
{
    return super::initialize(_c2, IIRF::Second,
			     _c0, IIRF::Zeroth)(ib, ie, out);
}

//! Gauss核による縦横両方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::diffHV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c1, IIRF::First,
			     _c1, IIRF::First)(ib, ie, out);
}

//! Gauss核による縦方向2階微分
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
  \return	出力2次元データ配列の末尾の次の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline OUT
GaussianConvolver2<T>::diffVV(IN ib, IN ie, OUT out)
{
    return super::initialize(_c0, IIRF::Zeroth,
			     _c2, IIRF::Second)(ib, ie, out);
}

}
#endif	/* !__TUGaussianConvolver_h */
