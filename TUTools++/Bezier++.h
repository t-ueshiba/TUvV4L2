/*
 *  $Id: Bezier++.h,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#ifndef __TUBezierPP_h
#define __TUBezierPP_h

#include "TU/Geometry++.h"

namespace TU
{
/************************************************************************
*  class BezierCurveBase<T, C>						*
************************************************************************/
template <class T, class C>
class BezierCurveBase : private Array<C>
{
  public:
    BezierCurveBase(u_int p=0)		:Array<C>(p+1)		{}
    BezierCurveBase(const Array<C>& b)	:Array<C>(b)		{}

    static u_int	dim()			{return C::dim();}

    const C&	operator [](int i)	  const	{return
						   Array<C>::operator [](i);}
    C&		operator [](int i)		{return
						   Array<C>::operator [](i);}
    u_int	degree()		  const	{return Array<C>::dim()-1;}
    C		operator ()(T t)	  const	;
    Array<C>	deCasteljau(T t, u_int r) const	;
    void	elevateDegree()			;
    
  //    Array<C>::operator [];
    Array<C>::operator ==;
    Array<C>::operator !=;
    Array<C>::save;
    Array<C>::restore;

    friend class Array2<BezierCurveBase<T, C> >; // allow access to resize.
    
    friend std::istream&
    operator >>(std::istream& in, BezierCurveBase<T, C>& b)
	{return in >> (Array<C>&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BezierCurveBase<T, C>& b)
	{return out << (const Array<C>&)b;}
};

/************************************************************************
*  class BezierCurve<T, D>						*
************************************************************************/
template <class T, u_int D>
class BezierCurve : public BezierCurveBase<T, Coordinate<T, D> >
{
  public:
    typedef Coordinate<T, D>	Coord;
    
    BezierCurve(u_int p) :BezierCurveBase<T, Coord>(p)			{}
    BezierCurve(const Array<Coord>& b)
	:BezierCurveBase<T, Coord>(b)					{}

    operator T*()			const	{return (*this)[0];}
};

/************************************************************************
*  class RationalBezierCurve<T, D>					*
************************************************************************/
template <class T, u_int D>
class RationalBezierCurve : public BezierCurveBase<T, CoordinateP<T, D> >
{
  public:
    typedef CoordinateP<T, D>	Coord;

    RationalBezierCurve(u_int p) :BezierCurveBase<T, Coord>(p)		{}
    RationalBezierCurve(const Array<Coord>& b)
	:BezierCurveBase<T, Coord>(b)					{}

    operator T*()			const	{return (*this)[0];}
};

/************************************************************************
*  class BezierSurfaceBase<T, C>					*
************************************************************************/
template <class T, class C>
class BezierSurfaceBase : protected Array2<BezierCurveBase<T, C> >
{
  public:
    typedef BezierCurveBase<T, C>	Curve;

    BezierSurfaceBase(u_int p, u_int q) :Array2<Curve>(q+1, p+1)	{}
    BezierSurfaceBase(const Array2<Array<C> >& b)			;
    
    static u_int	dim()				{return C::dim();}

    const Curve&	operator [](int i)	const
				{return Array2<Curve>::operator [](i);}
    Curve&		operator [](int i)		
				{return Array2<Curve>::operator [](i);}
    u_int	uDegree()			const	{return ncol()-1;}
    u_int	vDegree()			const	{return nrow()-1;}
    C		operator ()(T u, T v)		const	;
    Array2<Array<C> >
		deCasteljau(T u, T v, u_int r)	const	;
    void	uElevateDegree()			;
    void	vElevateDegree()			;

  //    Array2<Curve>::operator [];
    Array2<Curve>::operator ==;
    Array2<Curve>::operator !=;
    Array2<Curve>::save;
    Array2<Curve>::restore;
    
    friend std::istream&
    operator >>(std::istream& in, BezierSurfaceBase<T, C>& b)
	{return in >> (Array2<Curve>&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BezierSurfaceBase<T, C>& b)
	{return out << (const Array2<Curve>&)b;}
};

/************************************************************************
*  class BezierSurface<T>						*
************************************************************************/
template <class T>
class BezierSurface : public BezierSurfaceBase<T, TU::Coordinate<T, 3u> >
{
  public:
    typedef Coordinate<T, 3u>		Coord;
    
    BezierSurface(u_int p, u_int q)
	:BezierSurfaceBase<T, Coord>(p, q)			{}
    BezierSurface(const Array2<Array<Coord> >& b)
	:BezierSurfaceBase<T, Coord>(b)				{}

    operator T*()			const	{return (*this)[0][0];}
};

/************************************************************************
*  class RationalBezierSurface<T>					*
************************************************************************/
template <class T>
class RationalBezierSurface
	: public BezierSurfaceBase<T, TU::CoordinateP<T, 3u> >
{
  public:
    typedef CoordinateP<T, 3u>	Coord;

    RationalBezierSurface(u_int p, u_int q)
	:BezierSurfaceBase<T, Coord>(p)				{}
    RationalBezierSurface(const Array2<Array<Coord> >& b)
	:BezierSurfaceBase<T, Coord>(b)				{}

    operator T*()			const	{return (*this)[0][0];}
};
 
}

#endif
