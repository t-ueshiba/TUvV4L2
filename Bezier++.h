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
 *  $Id: Bezier++.h,v 1.12 2009-07-09 04:18:48 ueshiba Exp $
 */
#ifndef __TUBezierPP_h
#define __TUBezierPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BezierCurve<C>							*
************************************************************************/
template <class C>
class BezierCurve : private Array<C>
{
  public:
    typedef C					value_type;
    typedef typename value_type::value_type	T;
    
    BezierCurve(u_int p=0)	 :Array<C>(p+1)	{}
    BezierCurve(const Array<C>& b) :Array<C>(b)	{}

    static u_int	dim()			{return C::size();}

    u_int	degree()		  const	{return Array<C>::dim()-1;}
    C		operator ()(T t)	  const	;
    Array<C>	deCasteljau(T t, u_int r) const	;
    void	elevateDegree()			;
		operator const T*()	  const	{return (*this)[0];}

    friend	class Array2<BezierCurve<C> >;	// allow access to resize.
    
    using	Array<C>::operator [];
    using	Array<C>::operator ==;
    using	Array<C>::operator !=;
    using	Array<C>::save;
    using	Array<C>::restore;

    friend std::istream&
    operator >>(std::istream& in, BezierCurve<C>& b)
	{return in >> (Array<C>&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BezierCurve<C>& b)
	{return out << (const Array<C>&)b;}
};

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
template <class C>
class BezierSurface : private Array2<BezierCurve<C> >
{
  public:
    typedef C						value_type;
    typedef typename value_type::value_type		T;
    typedef BezierCurve<C>				Curve;

    BezierSurface(u_int p, u_int q) :Array2<Curve>(q+1, p+1)	{}
    BezierSurface(const Array2<Array<C> >& b)			;

    static u_int	dim()				{return C::size();}

    u_int	uDegree()			const	{return ncol()-1;}
    u_int	vDegree()			const	{return nrow()-1;}
    C		operator ()(T u, T v)		const	;
    Array2<Array<C> >
		deCasteljau(T u, T v, u_int r)	const	;
    void	uElevateDegree()			;
    void	vElevateDegree()			;
		operator const T*()		const	{return (*this)[0][0];}

    using	Array2<Curve>::operator [];
    using	Array2<Curve>::nrow;
    using	Array2<Curve>::ncol;
    using	Array2<Curve>::operator ==;
    using	Array2<Curve>::operator !=;
    using	Array2<Curve>::save;
    using	Array2<Curve>::restore;
    
    friend std::istream&
    operator >>(std::istream& in, BezierSurface<C>& b)
	{return in >> (Array2<Curve>&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BezierSurface<C>& b)
	{return out << (const Array2<Curve>&)b;}
};

typedef BezierSurface<Vector3f>	BezierSurface3f;
typedef BezierSurface<Vector4f>	RationalBezierSurface3f;
typedef BezierSurface<Vector3d>	BezierSurface3d;
typedef BezierSurface<Vector4d>	RationalBezierSurface3d;
 
}

#endif
