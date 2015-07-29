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
  \file		Bezier++.h
  \brief	Bezier曲線およびBezier曲面に関連するクラスの定義と実装
*/
#ifndef __TU_BEZIERPP_H
#define __TU_BEZIERPP_H

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BezierCurve<C>							*
************************************************************************/
//! 非有理または有理Bezier曲線を表すクラス
/*!
  \param C	制御点座標の型．d次元空間中の非有理曲線であればd次元ベクトル，
		有理曲線であれば(d+1)次元ベクトル．
*/
template <class C>
class BezierCurve : private Array<C>
{
  public:
    typedef Array<C>				coord_array;
    typedef C					coord_type;
    typedef coord_type				value_type;
    typedef typename coord_type::element_type	element_type;
    
  private:
    typedef coord_array				super;

  public:
  //! 指定した次数のBezier曲線を作る．
  /*!
    \param p	次数
  */
    BezierCurve(size_t p=0)	:super(p+1)	{}

  //! 指定した制御点を持つBezier曲線を作る．
  /*!
    \param b	サイズが(次数+1)個である制御点の1次元配列
  */
    BezierCurve(const coord_array& b) :super(b)	{}

  //! 曲線が属す空間の次元を調べる．
  /*!
    \return	空間の次元
  */
    static size_t	dim()			{return coord_type::size();}

  //! 曲線の次数(= 制御点数-1)を調べる．
  /*!
    \return	次数
  */
    size_t	degree()		const	{return super::size()-1;}

    coord_type	operator ()(element_type t)		const	;
    coord_array	deCasteljau(element_type t, size_t r)	const	;
    void	elevateDegree()					;

  //! 制御点の1次元配列へのポインタを返す．
  /*!
    \return	制御点の配列へのポインタ
  */
    const element_type*	data()		const	{return (*this)[0].data();}

    friend	class Array2<BezierCurve>;	// allow access to resize.
    
    using	super::operator [];
    using	super::operator ==;
    using	super::operator !=;
    using	super::save;
    using	super::restore;

  //! ストリームからBezier曲線を読み込む．
  /*!
    \param in	入力ストリーム
    \param b	Bezier曲線
    \return	inで指定した入力ストリーム
  */
    friend std::istream&
    operator >>(std::istream& in, BezierCurve& b)
	{return in >> (super&)b;}

  //! ストリームにBezier曲線を書き出す．
  /*!
    \param out	出力ストリーム
    \param b	Bezier曲線
    \return	outで指定した出力ストリーム
  */
    friend std::ostream&
    operator <<(std::ostream& out, const BezierCurve& b)
	{return out << (const super&)b;}
};

//! 指定したパラメータ値に対応する曲線上の点を調べる．
/*!
  \param t	曲線上の位置を指定するパラメータ値
  \return	パラメータ値に対応する曲線上の点
*/
template <class C> typename BezierCurve<C>::coord_type
BezierCurve<C>::operator ()(element_type t) const
{
    element_type	s = 1.0 - t, fact = 1.0;
    int			nCi = 1;
    coord_type	b((*this)[0] * s);
    for (size_t i = 1; i < degree(); ++i)
    {
	fact *= t;
      /* 
       * Be careful! We cannot use operator "*=" here, because operator "/"
       * must not produce remainder
       */
	nCi = nCi * (degree() - i + 1) / i;
	(b += fact * nCi * (*this)[i]) *= s;
    }
    b += fact * t * (*this)[degree()];
    return b;
}

//! de Casteljauアルゴリズムを実行する．
/*!
  \param t	曲線上の位置を指定するパラメータ値
  \param r
*/
template <class C> Array<C>
BezierCurve<C>::deCasteljau(element_type t, size_t r) const
{
    if (r > degree())
	r = degree();

    const element_type	s = 1.0 - t;
    Array<coord_type>	b_tmp(*this);
    for (size_t k = 1; k <= r; ++k)
	for (size_t i = 0; i <= degree() - k; ++i)
	    (b_tmp[i] *= s) += t * b_tmp[i+1];
    b_tmp.resize(degree() - r + 1);
    return b_tmp;
}

//! 曲線の形状を変えずに次数を1だけ上げる．
template <class C> void
BezierCurve<C>::elevateDegree()
{
    coord_array	b_tmp(*this);
    super::resize(degree() + 2);
    (*this)[0] = b_tmp[0];
    for (size_t i = 1; i < degree(); ++i)
    {
	element_type	alpha = element_type(i) / element_type(degree());
	
	(*this)[i] = alpha * b_tmp[i-1] + (1.0 - alpha) * b_tmp[i];
    }
    (*this)[degree()] = b_tmp[degree()-1];
}

typedef BezierCurve<Vector2f>	BezierCurve2f;
typedef BezierCurve<Vector3f>	RationalBezierCurve2f;
typedef BezierCurve<Vector3f>	BezierCurve3f;
typedef BezierCurve<Vector4f>	RationalBezierCurve3f;
typedef BezierCurve<Vector2d>	BezierCurve2d;
typedef BezierCurve<Vector3d>	RationalBezierCurve2d;
typedef BezierCurve<Vector3d>	BezierCurve3d;
typedef BezierCurve<Vector4d>	RationalBezierCurve3d;

/************************************************************************
*  class BezierSurface<C>						*
************************************************************************/
//! 非有理または有理Bezier曲面を表すクラス
/*!
  \param C	制御点の型．d次元空間中の非有理曲面であればd次元ベクトル，
		有理曲面であれば(d+1)次元ベクトル．
*/
template <class C>
class BezierSurface : private Array2<BezierCurve<C> >
{
  public:
    typedef C						coord_type;
    typedef BezierCurve<coord_type>			curve_type;
    typedef Array2<Array<coord_type> >			coord_array2;
    typedef typename coord_type::element_type		element_type;

  private:
    typedef Array2<curve_type>				super;

  public:
  //! 指定した次数のBezier曲面を作る．
  /*!
    \param p	横方向次数
    \param q	縦方向次数
  */
    BezierSurface(size_t p, size_t q) :super(q+1, p+1)	{}

    BezierSurface(const coord_array2& b)		;

  //! 曲面が属す空間の次元を調べる．
  /*!
    \return	空間の次元
  */
    static size_t	dim()			{return coord_type::size();}

  //! 曲面の横方向次数を調べる．
  /*!
    \return	横方向次数
  */
    size_t	uDegree()		const	{return super::ncol()-1;}

  //! 曲面の縦方向次数を調べる．
  /*!
    \return	縦方向次数
  */
    size_t	vDegree()		const	{return super::nrow()-1;}

    coord_type	operator ()(element_type u, element_type v)	const	;
    coord_array2
		deCasteljau(element_type u, element_type v,
			    size_t r)				const	;
    void	uElevateDegree()					;
    void	vElevateDegree()					;

  //! 制御点の2次元配列へのポインタを返す．
  /*!
    \return	制御点の配列へのポインタ
  */
    const element_type*	data()		const	{return (*this)[0][0].data();}

    using	super::operator [];
    using	super::operator ==;
    using	super::operator !=;
    using	super::save;
    using	super::restore;
    
  //! ストリームからBezier曲面を読み込む．
  /*!
    \param in	入力ストリーム
    \param b	Bezier曲面
    \return	inで指定した入力ストリーム
  */
    friend std::istream&
    operator >>(std::istream& in, BezierSurface& b)
	{return in >> (super&)b;}

  //! ストリームにBezier曲面を書き出す．
  /*!
    \param out	出力ストリーム
    \param b	Bezier曲面
    \return	outで指定した出力ストリーム
  */
    friend std::ostream&
    operator <<(std::ostream& out, const BezierSurface& b)
	{return out << (const super&)b;}
};

//! 指定した制御点を持つBezier曲面を作る．
/*!
  \param b	サイズが(横方向次数+1)x(縦方向次数+1)である制御点の2次元配列
*/
template <class C>
BezierSurface<C>::BezierSurface(const coord_array2& b)
    :super(b.nrow(), b.ncol())
{
    for (size_t j = 0; j <= vDegree(); ++j)
	for (size_t i = 0; i <= uDegree(); ++i)
	    (*this)[j][i] = b[j][i];
}

//! 指定したパラメータ値に対応する曲面上の点を調べる．
/*!
  \param u	曲面上の位置を指定する横方向パラメータ値
  \param v	曲面上の位置を指定する縦方向パラメータ値
  \return	パラメータ値に対応する曲面上の点
*/
template <class C> typename BezierSurface<C>::coord_type
BezierSurface<C>::operator ()(element_type u, element_type v) const
{
    curve_type	vCurve(vDegree());
    for (size_t j = 0; j <= vDegree(); ++j)
	vCurve[j] = (*this)[j](u);
    return vCurve(v);
}
 
typedef BezierSurface<Vector3f>	BezierSurface3f;
typedef BezierSurface<Vector4f>	RationalBezierSurface3f;
typedef BezierSurface<Vector3d>	BezierSurface3d;
typedef BezierSurface<Vector4d>	RationalBezierSurface3d;
 
}
#endif	// !__TU_BEZIERPP_H
