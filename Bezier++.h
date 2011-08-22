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
 *  $Id: Bezier++.h,v 1.16 2011-08-22 00:06:25 ueshiba Exp $
 */
/*!
  \file		Bezier++.h
  \brief	Bezier曲線およびBezier曲面に関連するクラスの定義と実装
*/
#ifndef __TUBezierPP_h
#define __TUBezierPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BezierCurve<C>							*
************************************************************************/
//! 非有理または有理Bezier曲線を表すクラス
/*!
  \param C	制御点の型．d次元空間中の非有理曲線であればd次元ベクトル，
		有理曲線であれば(d+1)次元ベクトル．
*/
template <class C>
class BezierCurve : private Array<C>
{
  public:
    typedef C					value_type;
    typedef typename value_type::value_type	T;

  //! 指定した次数のBezier曲線を作る．
  /*!
    \param p	次数
  */
    BezierCurve(u_int p=0)	 :Array<C>(p+1)	{}

  //! 指定した制御点を持つBezier曲線を作る．
  /*!
    \param b	サイズが(次数+1)個である制御点の1次元配列
  */
    BezierCurve(const Array<C>& b) :Array<C>(b)	{}

  //! 曲線が属す空間の次元を調べる．
  /*!
    \return	空間の次元
  */
    static u_int	dim()			{return C::size();}

  //! 曲線の次数(= 制御点数-1)を調べる．
  /*!
    \return	次数
  */
    u_int	degree()		  const	{return Array<C>::dim()-1;}

    C		operator ()(T t)	  const	;
    Array<C>	deCasteljau(T t, u_int r) const	;
    void	elevateDegree()			;

  //! 制御点の1次元配列へのポインタを返す．
  /*!
    \return	制御点の配列へのポインタ
  */
		operator const T*()	  const	{return (*this)[0];}

    friend	class Array2<BezierCurve<C> >;	// allow access to resize.
    
    using	Array<C>::operator [];
    using	Array<C>::operator ==;
    using	Array<C>::operator !=;
    using	Array<C>::save;
    using	Array<C>::restore;

  //! ストリームからBezier曲線を読み込む．
  /*!
    \param in	入力ストリーム
    \param b	Bezier曲線
    \return	inで指定した入力ストリーム
  */
    friend std::istream&
    operator >>(std::istream& in, BezierCurve<C>& b)
	{return in >> (Array<C>&)b;}

  //! ストリームにBezier曲線を書き出す．
  /*!
    \param out	出力ストリーム
    \param b	Bezier曲線
    \return	outで指定した出力ストリーム
  */
    friend std::ostream&
    operator <<(std::ostream& out, const BezierCurve<C>& b)
	{return out << (const Array<C>&)b;}
};

//! 指定したパラメータ値に対応する曲線上の点を調べる．
/*!
  \param t	曲線上の位置を指定するパラメータ値
  \return	パラメータ値に対応する曲線上の点
*/
template <class C> C
BezierCurve<C>::operator ()(T t) const
{
    T		s = 1.0 - t, fact = 1.0;
    int		nCi = 1;
    C		b((*this)[0] * s);
    for (u_int i = 1; i < degree(); ++i)
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
BezierCurve<C>::deCasteljau(T t, u_int r) const
{
    if (r > degree())
	r = degree();

    const T	s = 1.0 - t;
    Array<C>	b_tmp(*this);
    for (u_int k = 1; k <= r; ++k)
	for (u_int i = 0; i <= degree() - k; ++i)
	    (b_tmp[i] *= s) += t * b_tmp[i+1];
    b_tmp.resize(degree() - r + 1);
    return b_tmp;
}

//! 曲線の形状を変えずに次数を1だけ上げる．
template <class C> void
BezierCurve<C>::elevateDegree()
{
    Array<C>	b_tmp(*this);
    Array<C>::resize(degree() + 2);
    (*this)[0] = b_tmp[0];
    for (u_int i = 1; i < degree(); ++i)
    {
	T	alpha = T(i) / T(degree());
	
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
    typedef C						value_type;
    typedef typename value_type::value_type		T;
    typedef BezierCurve<C>				Curve;

  //! 指定した次数のBezier曲面を作る．
  /*!
    \param p	横方向次数
    \param q	縦方向次数
  */
    BezierSurface(u_int p, u_int q) :Array2<Curve>(q+1, p+1)	{}

    BezierSurface(const Array2<Array<C> >& b)			;

  //! 曲面が属す空間の次元を調べる．
  /*!
    \return	空間の次元
  */
    static u_int	dim()				{return C::size();}

  //! 曲面の横方向次数を調べる．
  /*!
    \return	横方向次数
  */
    u_int	uDegree()			const	{return ncol()-1;}

  //! 曲面の縦方向次数を調べる．
  /*!
    \return	縦方向次数
  */
    u_int	vDegree()			const	{return nrow()-1;}

    C		operator ()(T u, T v)		const	;
    Array2<Array<C> >
		deCasteljau(T u, T v, u_int r)	const	;
    void	uElevateDegree()			;
    void	vElevateDegree()			;

  //! 制御点の2次元配列へのポインタを返す．
  /*!
    \return	制御点の配列へのポインタ
  */
		operator const T*()		const	{return (*this)[0][0];}

    using	Array2<Curve>::operator [];
    using	Array2<Curve>::nrow;
    using	Array2<Curve>::ncol;
    using	Array2<Curve>::operator ==;
    using	Array2<Curve>::operator !=;
    using	Array2<Curve>::save;
    using	Array2<Curve>::restore;
    
  //! ストリームからBezier曲面を読み込む．
  /*!
    \param in	入力ストリーム
    \param b	Bezier曲面
    \return	inで指定した入力ストリーム
  */
    friend std::istream&
    operator >>(std::istream& in, BezierSurface<C>& b)
	{return in >> (Array2<Curve>&)b;}

  //! ストリームにBezier曲面を書き出す．
  /*!
    \param out	出力ストリーム
    \param b	Bezier曲面
    \return	outで指定した出力ストリーム
  */
    friend std::ostream&
    operator <<(std::ostream& out, const BezierSurface<C>& b)
	{return out << (const Array2<Curve>&)b;}
};

//! 指定した制御点を持つBezier曲面を作る．
/*!
  \param b	サイズが(横方向次数+1)x(縦方向次数+1)である制御点の2次元配列
*/
template <class C>
BezierSurface<C>::BezierSurface(const Array2<Array<C> >& b)
    :Array2<Curve>(b.nrow(), b.ncol())
{
    for (u_int j = 0; j <= vDegree(); ++j)
	for (u_int i = 0; i <= uDegree(); ++i)
	    (*this)[j][i] = b[j][i];
}

//! 指定したパラメータ値に対応する曲面上の点を調べる．
/*!
  \param u	曲面上の位置を指定する横方向パラメータ値
  \param v	曲面上の位置を指定する縦方向パラメータ値
  \return	パラメータ値に対応する曲面上の点
*/
template <class C> C
BezierSurface<C>::operator ()(T u, T v) const
{
    BezierCurve<C>	vCurve(vDegree());
    for (u_int j = 0; j <= vDegree(); ++j)
	vCurve[j] = (*this)[j](u);
    return vCurve(v);
}
 
typedef BezierSurface<Vector3f>	BezierSurface3f;
typedef BezierSurface<Vector4f>	RationalBezierSurface3f;
typedef BezierSurface<Vector3d>	BezierSurface3d;
typedef BezierSurface<Vector4d>	RationalBezierSurface3d;
 
}

#endif
