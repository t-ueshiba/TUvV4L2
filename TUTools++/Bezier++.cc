/*
 *  $Id: Bezier++.cc,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#include "TU/Bezier++.h"

namespace TU
{
/************************************************************************
*  class BezierCurveBase<T, C>						*
************************************************************************/
template <class T, class C> C
BezierCurveBase<T, C>::operator ()(T t) const
{
    T		s = 1.0 - t, fact = 1.0;
    int		nCi = 1;
    C		b((*this)[0] * s);
    for (int i = 1; i < degree(); ++i)
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

template <class T, class C> Array<C>
BezierCurveBase<T, C>::deCasteljau(T t, u_int r) const
{
    if (r > degree())
	r = degree();

    const T	s = 1.0 - t;
    Array<C>	b_tmp(*this);
    for (int k = 1; k <= r; ++k)
	for (int i = 0; i <= degree() - k; ++i)
	    (b_tmp[i] *= s) += t * b_tmp[i+1];
    b_tmp.resize(degree() - r + 1);
    return b_tmp;
}

template <class T, class C> void
BezierCurveBase<T, C>::elevateDegree()
{
    Array<C>	b_tmp(*this);
    resize(degree() + 2);
    (*this)[0] = b_tmp[0];
    for (int i = 1; i < degree(); ++i)
    {
	T	alpha = T(i) / T(degree());
	
	(*this)[i] = alpha * b_tmp[i-1] + (1.0 - alpha) * b_tmp[i];
    }
    (*this)[degree()] = b_tmp[degree()-1];
}

/************************************************************************
*  class BezierSurfaceBase<T, C>					*
************************************************************************/
template <class T, class C>
BezierSurfaceBase<T, C>::BezierSurfaceBase(const Array2<Array<C> >& b)
    :Array2<Curve>(b.nrow(), b.ncol())
{
    for (int j = 0; j <= vDegree(); ++j)
	for (int i = 0; i <= uDegree(); ++i)
	    (*this)[j][i] = b[j][i];
}

template <class T, class C> C
BezierSurfaceBase<T, C>::operator ()(T u, T v) const
{
    BezierCurveBase<T, C>	vCurve(vDegree());
    for (int j = 0; j <= vDegree(); ++j)
	vCurve[j] = (*this)[j](u);
    return vCurve(v);
}
 
}
