/*
 *  $Id: Geometry++.cc,v 1.2 2002-07-25 02:38:04 ueshiba Exp $
 */
#include "TU/Geometry++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class CoordBase<T, D>						*
************************************************************************/
template <class T, u_int D>
CoordBase<T, D>::CoordBase(const CoordBase<T, D>& p)
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = p[i];
}

template <class T, u_int D>
CoordBase<T, D>::CoordBase(const Vector<double>& v)
{
    if (v.dim() != dim())
	throw std::invalid_argument("TU::CoordBase<T, D>::Coordinate(): dimension mismatch!");
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = v[i];
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator =(const CoordBase<T, D>& p)
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = p[i];
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator =(const Vector<double>& v)
{
    if (v.dim() != dim())
	throw std::invalid_argument("TU::CoordBase<T, D>::operator =(): dimension mismatch!");

    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = v[i];
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator +=(const CoordBase& p)
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] += p[i];
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator -=(const CoordBase& p)
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] -= p[i];
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator =(double c)
{
    for (u_int i = 0; i < dim(); )
	(*this)[i++] = c;
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator *=(double c)
{
    for (u_int i = 0; i < dim(); )
	(*this)[i++] *= c;
    return *this;
}

template <class T, u_int D> CoordBase<T, D>&
CoordBase<T, D>::operator /=(double c)
{
    for (u_int i = 0; i < dim(); )
	(*this)[i++] /= c;
    return *this;
}

template <class T, u_int D>
CoordBase<T, D>::operator Vector<double>() const
{
    Vector<double>	v(dim());
    for (u_int i = 0; i < v.dim(); ++i)
	v[i] = (*this)[i];
    return v;
}

template <class T, u_int D> int
CoordBase<T, D>::operator ==(const CoordBase<T, D>& p) const
{
    for (u_int i = 0; i < dim(); ++i)
	if ((*this)[i] != p[i])
	    return 0;
    return 1;
}

template <class T, u_int D> std::istream&
CoordBase<T, D>::restore(std::istream& in)
{
    in.read((char*)_p, sizeof(T) * dim());
    return in;
}

template <class T, u_int D> std::ostream&
CoordBase<T, D>::save(std::ostream& out) const
{
    out.write((const char*)_p, sizeof(T) * dim());
    return out;
}

template <class T, u_int D> CoordBase<T, D>
CoordBase<T, D>::normal() const
{
    CoordBase<T, D>	r(*this);
    r.normalize();
    return r;
}

template <class T, u_int D> void
CoordBase<T, D>::check_dim(u_int d) const
{
    if (dim() != d)
	throw std::invalid_argument("TUCoordBase<T, D>::check_dim: dimension mismatch!");
}

/*
 *  I/O functions
 */
template <class T, u_int D> std::istream&
operator >>(std::istream& in, CoordBase<T, D>& p)
{
    for (u_int i = 0; i < p.dim(); )
	in >> p[i++];
    return in >> std::ws;
}

template <class T, u_int D> std::ostream&
operator <<(std::ostream& out, const CoordBase<T, D>& p)
{
    for (u_int i = 0; i < p.dim(); )
	out << ' ' << p[i++];
    return out << std::endl;
}

/*
 *  numerical operators
 */
template <class T, u_int D> extern T
operator *(const CoordBase<T, D>& p, const CoordBase<T, D>& q)
{
    T	val = 0;
    for (int i = 0; i < D; ++i)
	val += p[i]*q[i];
    return val;
}

/************************************************************************
*  class Coordinate<T, D>						*
************************************************************************/
template <class T, u_int D>
Coordinate<T, D>::Coordinate(const CoordinateP<T, D>& p)
    :CoordBase<T, D>()
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = p[i] / p[dim()];
}

template <class T, u_int D> Coordinate<T, D>&
Coordinate<T, D>::operator =(const CoordinateP<T, D>& p)
{
    for (u_int i = 0; i < dim(); ++i)
	(*this)[i] = p[i] / p[dim()];
    return *this;
}

/*
 *  numerical operators
 */
template <class T> Coordinate<T, 3u>
operator ^(const Coordinate<T, 3u>& p, const Coordinate<T, 3u>& q)
{
    Coordinate<T, 3u>	val;
    val[0] = p[1] * q[2] - p[2] * q[1];
    val[1] = p[2] * q[0] - p[0] * q[2];
    val[2] = p[0] * q[1] - p[1] * q[0];
    return val;
}

/************************************************************************
*  class CoordinateP<T, D>						*
************************************************************************/
template <class T, u_int D>
CoordinateP<T, D>::CoordinateP(const Coordinate<T, D>& p)
    :CoordBase<T, D+1u>()
{
    u_int	i;
    
    for (i = 0; i < p.dim(); ++i)
	(*this)[i] = p[i];
    (*this)[i] = 1;
}

template <class T, u_int D> CoordinateP<T, D>&
CoordinateP<T, D>::operator =(const Coordinate<T, D>& p)
{
    u_int	i;
    
    for (i = 0; i < p.dim(); ++i)
	(*this)[i] = p[i];
    (*this)[i] = 1;
    return *this;
}

template <class T, u_int D> int
CoordinateP<T, D>::operator ==(const CoordinateP<T, D>& p) const
{
    u_int i0;

    for (i0 = 0; i0 < dim(); ++i0)
	if ((*this)[i0] != 0)
	    break;
    if (i0 == dim() || p[i0] == 0)
	return false;
    
    for (u_int i = 0; i < dim(); ++i)
	if ((*this)[i] * p[i0] != (*this)[i0] * p[i])
	    return false;
    return true;
}

/*
 *  numerical operators
 */
template <class T> CoordinateP<T, 2u>
operator ^(const CoordinateP<T, 2u>& p, const CoordinateP<T, 2u>& q)
{
    CoordinateP<T, 2u>	val;
    val[0] = p[1] * q[2] - p[2] * q[1];
    val[1] = p[2] * q[0] - p[0] * q[2];
    val[2] = p[0] * q[1] - p[1] * q[0];
    return val;
}

/************************************************************************
*  class Point2<T>							*
************************************************************************/
template <class T> Point2<T>&
Point2<T>::move(int dir)
{
    switch (dir % 8)
    {
      case 0:
	++(*this)[0];
	break;
      case 1:
      case -7:
	++(*this)[0];
	++(*this)[1];
	break;
      case 2:
      case -6:
	++(*this)[1];
	break;
      case 3:
      case -5:
	--(*this)[0];
	++(*this)[1];
	break;
      case 4:
      case -4:
	--(*this)[0];
	break;
      case 5:
      case -3:
	--(*this)[0];
	--(*this)[1];
	break;
      case 6:
      case -2:
	--(*this)[1];
	break;
      case 7:
      case -1:
	++(*this)[0];
	--(*this)[1];
	break;
    }
    return *this;
}

template <class T> int
Point2<T>::adj(const Point2<T>& p) const
{
    register int du = p[0] - (*this)[0], dv = p[1] - (*this)[1];

    if (du == 0 && dv == 0)
        return -1;
    switch (du)
    {
      case -1:
      case 0:
      case 1:
        switch (dv)
        {
          case -1:
          case 0:
          case 1:
            return 1;
          default:
            return 0;
        }
        break;
    }
    return 0;
}

template <class T> int
Point2<T>::dir(const Point2<T>& p) const
{
    register int du = p[0] - (*this)[0], dv = p[1] - (*this)[1];

    if (du == 0 && dv == 0)
        return 4;
    if (dv >= 0)
        if (du > dv)
            return 0;
        else if (du > 0)
            return 1;
        else if (du > -dv)
            return 2;
        else if (dv > 0)
            return 3;
        else
            return -4;
    else
        if (du >= -dv)
            return -1;
        else if (du >= 0)
            return -2;
        else if (du >= dv)
            return -3;
        else
            return -4;
}

template <class T> int
Point2<T>::angle(const Point2<T>& pp, const Point2<T>& pn) const
{
    int dp = pp.dir(*this), ang = dir(pn);
    
    if (dp == 4 || ang == 4)
        return 4;
    else if ((ang -= dp) > 3)
        return ang - 8;
    else if (ang < -4)
        return ang + 8;
    else
        return ang;
}
 
}
