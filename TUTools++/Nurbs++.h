/*
 *  $Id: Nurbs++.h,v 1.3 2006-12-19 07:09:24 ueshiba Exp $
 */
#ifndef __TUNurbsPP_h
#define __TUNurbsPP_h

#include "TU/Geometry++.h"

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
    
    const T&	operator [](int i)const	{return Array<T>::operator [](i);}
    T&		operator [](int i)	{return Array<T>::operator [](i);}
		operator T*()	const	{return Array<T>::operator T*();}
	
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

    using	Array<T>::dim;
  //using	Array<T>::operator [];		// knots
  //using	Array<T>::operator T*;
    
  private:
    u_int	_degree;
};

/************************************************************************
*  class BSplineCurveBase<T, C>						*
************************************************************************/
template <class T, class C>
class BSplineCurveBase : private Array<C>
{
  public:
    BSplineCurveBase(u_int degree, T us, T ue)		;

    static u_int	dim()			{return C::dim();}

    const C&	operator [](int i)const	{return Array<C>::operator [](i);}
    C&		operator [](int i)	{return Array<C>::operator [](i);}
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

  //    Array<C>::operator [];
    using	Array<C>::operator ==;
    using	Array<C>::operator !=;
    using	Array<C>::save;
    using	Array<C>::restore;

  private:
    BSplineKnots<T>	_knots;
};

/************************************************************************
*  class BSplineCurve<T, D>						*
************************************************************************/
template <class T, u_int D>
class BSplineCurve : public BSplineCurveBase<T, Coordinate<T, D> >
{
  public:
    typedef Coordinate<T, D>	Coord;
    
    BSplineCurve(u_int degree, T us=0.0, T ue=1.0)
	:BSplineCurveBase<T, Coord>(degree, us, ue)		{}

    operator T*()			const	{return (*this)[0];}
};

/************************************************************************
*  class RationalBSplineCurve<T, D>					*
************************************************************************/
template <class T, u_int D>
class RationalBSplineCurve
    : public BSplineCurveBase<T, CoordinateP<T, D> >
{
  public:
    typedef CoordinateP<T, D>	Coord;
    
    RationalBSplineCurve(u_int degree, T us=0.0, T ue=1.0)
	:BSplineCurveBase<T, Coord>(degree, us, ue)			{}

    operator T*()			const	{return (*this)[0];}
};

/************************************************************************
*  class BSplineSurfaceBase<T, C>					*
************************************************************************/
template <class T, class C>
class BSplineSurfaceBase : protected Array2<Array<C> >
{
  public:
    BSplineSurfaceBase(u_int uDegree, u_int vDegree,
			 T us, T ue, T vs, T ve)	;

    static u_int	dim()			{return C::dim();}

    const Array<C>&	operator [](int i)	const
			    {return Array2<Array<C> >::operator [](i);}
    Array<C>&		operator [](int i)
			    {return Array2<Array<C> >::operator [](i);}
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

  //using	Array2<Array<C> >::operator [];
    using	Array2<Array<C> >::ncol;
    using	Array2<Array<C> >::nrow;
    using	Array2<Array<C> >::operator ==;
    using	Array2<Array<C> >::operator !=;
    using	Array2<Array<C> >::save;
    using	Array2<Array<C> >::restore;
    
    friend std::istream&
    operator >>(std::istream& in, BSplineSurfaceBase<T, C>& b)
	{return in >> (Array2<Array<C> >&)b;}
    friend std::ostream&
    operator <<(std::ostream& out, const BSplineSurfaceBase<T, C>& b)
	{return out << (const Array2<Array<C> >&)b;}

  private:
    BSplineKnots<T>	_uKnots, _vKnots;
};

/************************************************************************
*  class BSplineSurface<T>						*
************************************************************************/
template <class T>
class BSplineSurface : public BSplineSurfaceBase<T, Coordinate<T, 3u> >
{
  public:
    typedef Coordinate<T, 3u>	Coord;
    
    BSplineSurface(u_int uDegree, u_int vDegree,
		     T us=0.0, T ue=1.0, T vs=0.0, T ve=1.0)
	:BSplineSurfaceBase<T, Coord>(uDegree, vDegree, us, ue, vs, ve) {}

    operator T*()			const	{return (*this)[0][0];}
};

}
#endif
