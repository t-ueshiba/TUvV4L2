/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: Nurbs++.h,v 1.8 2007-11-26 07:28:09 ueshiba Exp $
 */
#ifndef __TUNurbsPP_h
#define __TUNurbsPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BSplineKnots<T>						*
************************************************************************/
template <class T>
class BSplineKnots : private Array<T>
{
  public:
    BSplineKnots(u_int degree, T us, T ue)	;
    
    u_int	degree()		const	{return _degree;}
    u_int	M()			const	{return dim()-1;}
    u_int	L()			const	{return M()-degree()-degree();}
    int		findSpan(T u)		const	;
    int		leftmost(int k)		const	;
    int		rightmost(int k)	const	;
    u_int	multiplicity(int k)	const	;

    Array<T>	basis(T u, int& I)	const	;
    Array2<Array<T> >
      derivatives(T u, u_int K, int& I)	const	;

    int		insertKnot(T u)			;
    int		removeKnot(int k)		;
    void	elevateDegree()			{++_degree;}
		operator const T*()	const	{return
						  Array<T>::operator const T*();}

    using	Array<T>::dim;
    using	Array<T>::operator [];		// knots
    
  private:
    u_int	_degree;
};

/************************************************************************
*  class BSplineCurve<C>						*
************************************************************************/
template <class C>
class BSplineCurve : private Array<C>
{
  public:
    typedef typename C::ET			T;
    typedef typename Array<C>::ET		ET;
    typedef C					Coord;
    
    BSplineCurve(u_int degree, T us=0.0, T ue=1.0)	;

    static u_int	dim()			{return C::dim();}

    u_int	degree()		  const	{return _knots.degree();}
    u_int	M()			  const	{return _knots.M();}
    u_int	L()			  const	{return _knots.L();}
    u_int	N()			  const	{return Array<C>::dim()-1;}
    T		knots(int i)		  const {return _knots[i];}
    u_int	multiplicity(int k)	  const {return
						     _knots.multiplicity(k);}
    const BSplineKnots<T>&
		knots()			  const	{return _knots;}

    C		operator ()(T u)	  const	;
    Array<C>	derivatives(T u, u_int K) const	;

    int		insertKnot(T u)			;
    int		removeKnot(int k)		;
    void	elevateDegree()			;
		operator const T*()	  const	{return (*this)[0];}

    using	Array<C>::operator [];
    using	Array<C>::operator ==;
    using	Array<C>::operator !=;
    using	Array<C>::save;
    using	Array<C>::restore;

  private:
    BSplineKnots<T>	_knots;
};

typedef BSplineCurve<Vector<float, FixedSizedBuf<float, 2> > >
BSplineCurve2f;
typedef BSplineCurve<Vector<float, FixedSizedBuf<float, 3> > >
RationalBSplineCurve2f;
typedef BSplineCurve<Vector<float, FixedSizedBuf<float, 3> > >
BSplineCurve3f;
typedef BSplineCurve<Vector<float, FixedSizedBuf<float, 4> > >
RationalBSplineCurve3f;
typedef BSplineCurve<Vector<double, FixedSizedBuf<double, 2> > >
BSplineCurve2d;
typedef BSplineCurve<Vector<double, FixedSizedBuf<double, 3> > >
RationalBSplineCurve2d;
typedef BSplineCurve<Vector<double, FixedSizedBuf<double, 3> > >
BSplineCurve3d;
typedef BSplineCurve<Vector<double, FixedSizedBuf<double, 4> > >
RationalBSplineCurve3d;
    
/************************************************************************
*  class BSplineSurface<C>						*
************************************************************************/
template <class C>
class BSplineSurface : private Array2<Array<C> >
{
  public:
    typedef typename C::ET			T;
    typedef typename Array2<Array<C> >::ET	ET;
    typedef C					Coord;
    
    BSplineSurface(u_int uDegree, u_int vDegree,
		   T us=0.0, T ue=1.0, T vs=0.0, T ve=1.0)	;

    static u_int	dim()			{return C::dim();}

    u_int	uDegree()		const	{return _uKnots.degree();}
    u_int	uM()			const	{return _uKnots.M();}
    u_int	uL()			const	{return _uKnots.L();}
    u_int	uN()			const	{return ncol()-1;}
    u_int	vDegree()		const	{return _vKnots.degree();}
    u_int	vM()			const	{return _vKnots.M();}
    u_int	vL()			const	{return _vKnots.L();}
    u_int	vN()			const	{return nrow()-1;}
    T		uKnots(int i)		const	{return _uKnots[i];}
    T		vKnots(int j)		const	{return _vKnots[j];}
    u_int	uMultiplicity(int k)	const	{return
						     _uKnots.multiplicity(k);}
    u_int	vMultiplicity(int l)	const	{return
						     _vKnots.multiplicity(l);}
    const BSplineKnots<T>&
		uKnots()		const	{return _uKnots;}
    const BSplineKnots<T>&
		vKnots()		const	{return _vKnots;}

    C		operator ()(T u, T v)	const	;
    Array2<Array<C> >
	derivatives(T u, T v, u_int D)	const	;

    int		uInsertKnot(T u)		;
    int		vInsertKnot(T v)		;
    int		uRemoveKnot(int k)		;
    int		vRemoveKnot(int l)		;
    void	uElevateDegree()		;
    void	vElevateDegree()		;
		operator const T*()	const	{return (*this)[0][0];}

    using	Array2<Array<C> >::operator [];
    using	Array2<Array<C> >::ncol;
    using	Array2<Array<C> >::nrow;
    using	Array2<Array<C> >::operator ==;
    using	Array2<Array<C> >::operator !=;
    using	Array2<Array<C> >::save;
    using	Array2<Array<C> >::restore;
    
    friend std::istream&
    operator >>(std::istream& in, BSplineSurface<C>& b)
	{return in >> (Array2<Array<C> >&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BSplineSurface<C>& b)
	{return out << (const Array2<Array<C> >&)b;}

  private:
    BSplineKnots<T>	_uKnots, _vKnots;
};

typedef BSplineSurface<Vector<float, FixedSizedBuf<float, 3> > >
BSplineSurface3f;
typedef BSplineSurface<Vector<float, FixedSizedBuf<float, 4> > >
RationalBSplineSurface3f;
typedef BSplineSurface<Vector<double, FixedSizedBuf<double, 3> > >
BSplineSurface3d;
typedef BSplineSurface<Vector<double, FixedSizedBuf<double, 4> > >
RationalBSplineSurface3d;

}
#endif
