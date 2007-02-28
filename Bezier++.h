/*
 *  $Id: Bezier++.h,v 1.5 2007-02-28 00:16:06 ueshiba Exp $
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
    typedef typename C::value_type		T;
    typedef C					Coord;
    
    BezierCurve(u_int p=0)		:Array<C>(p+1)		{}
    BezierCurve(const Array<C>& b)	:Array<C>(b)		{}

    static u_int	dim()			{return C::dim();}

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
class BezierSurface : protected Array2<BezierCurve<C> >
{
  public:
    typedef BezierCurve<C>	Curve;
    typedef typename Curve::T	T;

    BezierSurface(u_int p, u_int q) :Array2<Curve>(q+1, p+1)	{}
    BezierSurface(const Array2<Array<C> >& b)			;

    static u_int	dim()				{return C::dim();}

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
