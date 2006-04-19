/*
 *  平成9年 電子技術総合研究所 植芝俊夫 著作権所有
 *
 *  著作者による許可なしにこのプログラムの第三者への開示、複製、改変、
 *  使用等その他の著作人格権を侵害する行為を禁止します。
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *
 *  Copyright 1996
 *  Toshio UESHIBA, Electrotechnical Laboratory
 *
 *  All rights reserved.
 *  Any changing, copying or giving information about source programs of
 *  any part of this software and/or documentation without permission of the
 *  authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damage in use of this program.
 */

/*
 *  $Id: Vector++.cc,v 1.12 2006-04-19 02:34:37 ueshiba Exp $
 */
#include "TU/Vector++.h"
#include <stdexcept>
#include <iomanip>

namespace TU
{
template <class T> inline void
swap(T& a, T& b)
{
    const T tmp = a;
    a = b;
    b = tmp;
}

/************************************************************************
*  class Vector<T>							*
************************************************************************/
template <class T> const Vector<T>
Vector<T>::operator ()(u_int i, u_int dd) const	// partial vector
{
    return Vector<T>(*this, i, dd);
}

template <class T> Vector<T>
Vector<T>::operator ()(u_int i, u_int dd)	// partial vector
{
    return Vector<T>(*this, i, dd);
}

template <class T> Vector<T>&
Vector<T>::operator ^=(const Vector<T>& v)	// outer product
{
    check_dim(v.dim());
    if (dim() != 3)
	throw std::invalid_argument("TU::Vector<T>::operator ^=: dimension must be 3");
    Vector<T> tmp(*this);
    (*this)[0] = tmp[1] * v[2] - tmp[2] * v[1];
    (*this)[1] = tmp[2] * v[0] - tmp[0] * v[2];
    (*this)[2] = tmp[0] * v[1] - tmp[1] * v[0];
    return *this;
}

template <class T> Vector<T>
Vector<T>::normal() const			// return normalized vector
{
    Vector<T> r(*this);
    r.normalize();
    return r;
}

template <class T> Vector<T>&
Vector<T>::solve(const Matrix<T>& m)
{
    LUDecomposition<T>(m).substitute(*this);
    return *this;
}

template <class T> Matrix<T>
Vector<T>::skew() const
{
    if (dim() != 3)
	throw std::invalid_argument("TU::Vector<T>::skew: dimension must be 3");
    Matrix<T>	r(3, 3);
    r[2][1] = (*this)[0];
    r[0][2] = (*this)[1];
    r[1][0] = (*this)[2];
    r[1][2] = -r[2][1];
    r[2][0] = -r[0][2];
    r[0][1] = -r[1][0];
    return r;
}

/************************************************************************
*  class Matrix<T>							*
************************************************************************/
template <class T> const Matrix<T>
Matrix<T>::operator ()(u_int i, u_int j, u_int r, u_int c) const
{
    return Matrix<T>(*this, i, j, r, c);
}

template <class T> Matrix<T>
Matrix<T>::operator ()(u_int i, u_int j, u_int r, u_int c)
{
    return Matrix<T>(*this, i, j, r, c);
}

template <class T> Matrix<T>&
Matrix<T>::diag(double c)
{
    check_dim(dim());
    *this = 0.0;
    for (int i = 0; i < dim(); i++)
	(*this)[i][i] = c;
    return *this;
}

template <class T> Matrix<T>&
Matrix<T>::rot(double angle, int axis)
{
    check_dim(dim());
    return *this;
}

template <class T> Matrix<T>&
Matrix<T>::operator ^=(const Vector<T>& v)
{
    for (int i = 0; i < nrow(); i++)
	(*this)[i] ^= v;
    return *this;
}

template <class T> Matrix<T>
Matrix<T>::trns() const				// transpose
{
    Matrix<T> val(ncol(), nrow());
    for (int i = 0; i < nrow(); i++)
	for (int j = 0; j < ncol(); j++)
	    val[j][i] = (*this)[i][j];
    return val;
}

template <class T> Matrix<T>
Matrix<T>::inv() const
{
    Matrix<T>	identity(nrow(), ncol());
    
    return identity.diag(1.0).solve(*this);
}

template <class T> Matrix<T>&
Matrix<T>::solve(const Matrix<T>& m)
{
    LUDecomposition<T>	lu(m);
    
    for (int i = 0; i < nrow(); i++)
	lu.substitute((*this)[i]);
    return *this;
}

template <class T> T
Matrix<T>::det() const
{
    return LUDecomposition<T>(*this).det();
}

template <class T> T
Matrix<T>::det(u_int p, u_int q) const
{
    Matrix<T>		d(nrow()-1, ncol()-1);
    for (int i = 0; i < p; ++i)
    {
	for (int j = 0; j < q; ++j)
	    d[i][j] = (*this)[i][j];
	for (int j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i][j+1];
    }
    for (int i = p; i < d.nrow(); ++i)
    {
	for (int j = 0; j < q; ++j)
	    d[i][j] = (*this)[i+1][j];
	for (int j = q; j < d.ncol(); ++j)
	    d[i][j] = (*this)[i+1][j+1];
    }
    return d.det();
}

template <class T> T
Matrix<T>::trace() const
{
    if (nrow() != ncol())
        throw
	  std::invalid_argument("TU::Matrix<T>::trace(): not square matrix!!");
    T	val = 0.0;
    for (int i = 0; i < nrow(); ++i)
	val += (*this)[i][i];
    return val;
}

template <class T> Matrix<T>
Matrix<T>::adj() const
{
    Matrix<T>		val(nrow(), ncol());
    for (int i = 0; i < val.nrow(); ++i)
	for (int j = 0; j < val.ncol(); ++j)
	    val[i][j] = ((i + j) % 2 ? -det(j, i) : det(j, i));
    return val;
}

template <class T> Matrix<T>
Matrix<T>::pinv(double cndnum) const
{
    SVDecomposition<T>	svd(*this);
    Matrix<T>		B(svd.ncol(), svd.nrow());
    
    for (int i = 0; i < svd.diagonal().dim(); ++i)
	if (fabs(svd[i]) * cndnum > fabs(svd[0]))
	    B += (svd.Ut()[i] / svd[i]) % svd.Vt()[i];

    return B;
}

template <class T> Matrix<T>
Matrix<T>::eigen(Vector<T>& eval) const
{
    TriDiagonal<T>	tri(*this);

    tri.diagonalize();
    eval = tri.diagonal();

    return tri.Ut();
}

template <class T> Matrix<T>
Matrix<T>::geigen(const Matrix<T>& B, Vector<T>& eval) const
{
    Matrix<T>	Ltinv = B.cholesky().inv(), Linv = Ltinv.trns();
    Matrix<T>	Ut = (Linv * (*this) * Ltinv).eigen(eval);
    
    return Ut * Linv;
}

template <class T> Matrix<T>
Matrix<T>::cholesky() const
{
    if (nrow() != ncol())
        throw
	    std::invalid_argument("TU::Matrix<T>::cholesky(): not square matrix!!");

    Matrix<T>	Lt(*this);
    for (int i = 0; i < nrow(); ++i)
    {
	T d = Lt[i][i];
	if (d <= 0.0)
	    throw std::runtime_error("TU::Matrix<T>::cholesky(): not positive definite matrix!!");
	for (int j = 0; j < i; ++j)
	    Lt[i][j] = 0.0;
	Lt[i][i] = d = sqrt(d);
	for (int j = i + 1; j < ncol(); ++j)
	    Lt[i][j] /= d;
	for (int j = i + 1; j < nrow(); ++j)
	    for (int k = j; k < ncol(); ++k)
		Lt[j][k] -= (Lt[i][j] * Lt[i][k]);
    }
    
    return Lt;
}

template <class T> Matrix<T>&
Matrix<T>::normalize()
{
    double	sum = 0.0;
    
    for (int i = 0; i < nrow(); ++i)
	sum += (*this)[i] * (*this)[i];
    return *this /= sqrt(sum);
}

template <class T> Matrix<T>&
Matrix<T>::rotate_from_left(const Rotation& r)
{
    for (int j = 0; j < ncol(); j++)
    {
	const T	tmp = (*this)[r.p()][j];
	
	(*this)[r.p()][j] =  r.cos()*tmp + r.sin()*(*this)[r.q()][j];
	(*this)[r.q()][j] = -r.sin()*tmp + r.cos()*(*this)[r.q()][j];
    }
    return *this;
}

template <class T> Matrix<T>&
Matrix<T>::rotate_from_right(const Rotation& r)
{
    for (int i = 0; i < nrow(); i++)
    {
	const T	tmp = (*this)[i][r.p()];
	
	(*this)[i][r.p()] =  tmp*r.cos() + (*this)[i][r.q()]*r.sin();
	(*this)[i][r.q()] = -tmp*r.sin() + (*this)[i][r.q()]*r.cos();
    }
    return *this;
}

template <class T> double
Matrix<T>::square() const
{
    double	val = 0.0;
    for (int i = 0; i < nrow(); ++i)
	val += (*this)[i] * (*this)[i];
    return val;
}

template <class T> Matrix<T>&
Matrix<T>::symmetrize()
{
    for (int i = 0; i < nrow(); ++i)
	for (int j = 0; j < i; ++j)
	    (*this)[j][i] = (*this)[i][j];
    return *this;
}

template <class T> Matrix<T>&
Matrix<T>::antisymmetrize()
{
    for (int i = 0; i < nrow(); ++i)
    {
	(*this)[i][i] = 0.0;
	for (int j = 0; j < i; ++j)
	    (*this)[j][i] = -(*this)[i][j];
    }
    return *this;
}

template <class T> void
Matrix<T>::rot2angle(double& theta_x, double& theta_y, double& theta_z) const
{
    using namespace	std;
    
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2angle: input matrix must be 3x3!!");

    if ((*this)[0][0] == 0.0 && (*this)[0][1] == 0.0)
    {
	theta_x = atan2(-(*this)[2][1], (*this)[1][1]);
	theta_y = ((*this)[0][2] < 0.0 ? M_PI / 2.0 : -M_PI / 2.0);
	theta_z = 0.0;
    }
    else
    {
	theta_x = atan2((*this)[1][2], (*this)[2][2]);
	theta_y = -asin((*this)[0][2]);
	theta_z = atan2((*this)[0][1], (*this)[0][0]);
    }
}

template <class T> Vector<T>
Matrix<T>::rot2axis(double& c, double& s) const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

  // Compute cosine and sine of rotation angle.
    const double	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2];
    c = (trace - 1.0) / 2.0;
    s = sqrt((trace + 1.0)*(3.0 - trace)) / 2.0;

  // Compute rotation axis.
    Vector<T>	n(3);
    n[0] = (*this)[1][2] - (*this)[2][1];
    n[1] = (*this)[2][0] - (*this)[0][2];
    n[2] = (*this)[0][1] - (*this)[1][0];
    n.normalize();

    return n;
}

template <class T> Vector<T>
Matrix<T>::rot2axis() const
{
    if (nrow() != 3 || ncol() != 3)
	throw std::invalid_argument("TU::Matrix<T>::rot2axis: input matrix must be 3x3!!");

    Vector<T>	axis(3);
    axis[0] = ((*this)[1][2] - (*this)[2][1]) * 0.5;
    axis[1] = ((*this)[2][0] - (*this)[0][2]) * 0.5;
    axis[2] = ((*this)[0][1] - (*this)[1][0]) * 0.5;
    const double	s = sqrt(axis.square());
    if (s + 1.0 == 1.0)		// s << 1 ?
	return axis;
    const double	trace = (*this)[0][0] + (*this)[1][1] + (*this)[2][2];
    if (trace > 1.0)		// cos > 0 ?
	return  asin(s) / s * axis;
    else
	return -asin(s) / s * axis;
}

template <class T> Matrix<T>
Matrix<T>::Rt(const Vector<T>& n, double c, double s)
{
    Matrix<T>	Qt = n % n;
    Qt *= (1.0 - c);
    Qt[0][0] += c;
    Qt[1][1] += c;
    Qt[2][2] += c;
    Qt[0][1] += n[2] * s;
    Qt[0][2] -= n[1] * s;
    Qt[1][0] -= n[2] * s;
    Qt[1][2] += n[0] * s;
    Qt[2][0] += n[1] * s;
    Qt[2][1] -= n[0] * s;

    return Qt;
}

template <class T> Matrix<T>
Matrix<T>::Rt(const Vector<T>& axis)
{
    double	theta = axis.length();
    if (theta + 1.0 == 1.0)		// theta << 1 ?
	return I(3);
    else
    {
	double	c = cos(theta), s = sin(theta);
	return Rt(axis / theta, c, s);
    }
}

/************************************************************************
*  numerical operators							*
************************************************************************/
template <class T> double
operator *(const Vector<T>& v, const Vector<T>& w)	// inner product
{
    v.check_dim(w.dim());
    double val = 0;
    for (int i = 0; i < v.dim(); i++)
	val += v[i] * w[i];
    return val;
}

template <class T> Vector<T>
operator *(const Vector<T>& v, const Matrix<T>& m)	// multiply by matrix
{
    v.check_dim(m.nrow());
    Vector<T> val(m.ncol());
    for (int j = 0; j < m.ncol(); j++)
	for (int i = 0; i < m.nrow(); i++)
	    val[j] += v[i] * m[i][j];
    return val;
}

template <class T> Matrix<T>
operator %(const Vector<T>& v, const Vector<T>& w)	// multiply by vector
{
    Matrix<T> val(v.dim(), w.dim());
    for (int i = 0; i < v.dim(); i++)
	for (int j = 0; j < w.dim(); j++)
	    val[i][j] = v[i] * w[j];
    return val;
}

template <class T> Matrix<T>
operator ^(const Vector<T>& v, const Matrix<T>& m)
{
    v.check_dim(m.nrow());
    if (v.dim() != 3)
	throw std::invalid_argument("operator ^(const Vecotr<T>&, const Matrix<T>&): dimension of vector must be 3!!");
    Matrix<T>	val(m.nrow(), m.ncol());
    for (int j = 0; j < val.ncol(); j++)
    {
	val[0][j] = v[1] * m[2][j] - v[2] * m[1][j];
	val[1][j] = v[2] * m[0][j] - v[0] * m[2][j];
	val[2][j] = v[0] * m[1][j] - v[1] * m[0][j];
    }
    return val;
}

template <class T> Matrix<T>
operator *(const Matrix<T>& m, const Matrix<T>& n)	// multiply by matrix
{
    n.check_dim(m.ncol());
    Matrix<T> val(m.nrow(), n.ncol());
    for (int i = 0; i < m.nrow(); i++)
	for (int j = 0; j < n.ncol(); j++)
	    for (int k = 0; k < m.ncol(); k++)
		val[i][j] += m[i][k] * n[k][j];
    return val;
}

template <class T> Vector<T>
operator *(const Matrix<T>& m, const Vector<T>& v)	// multiply by vector
{
    Vector<T> val(m.nrow());
    for (int i = 0; i < m.nrow(); i++)
	val[i] = m[i] * v;
    return val;
}

/************************************************************************
*  class LUDecomposition<T>						*
************************************************************************/
template <class T>
LUDecomposition<T>::LUDecomposition(const Matrix<T>& m)
    :Array2<Vector<T> >(m), _index(ncol()), _det(1.0)
{
    if (nrow() != ncol())
        throw std::invalid_argument("TU::LUDecomposition<T>::LUDecomposition: not square matrix!!");

    for (int j = 0; j < ncol(); j++)	// initialize column index
	_index[j] = j;			// for explicit pivotting

    Vector<T>	scale(ncol());
    for (int j = 0; j < ncol(); j++)	// find maximum abs. value in each col.
    {					// for implicit pivotting
	double max = 0.0;

	for (int i = 0; i < nrow(); i++)
	{
	    const double tmp = fabs((*this)[i][j]);
	    if (tmp > max)
		max = tmp;
	}
	scale[j] = (max != 0.0 ? 1.0 / max : 1.0);
    }

    for (int i = 0; i < nrow(); i++)
    {
	for (int j = 0; j < i; j++)		// left part (j < i)
	{
	    T& sum = (*this)[i][j];
	    for (int k = 0; k < j; k++)
		sum -= (*this)[i][k] * (*this)[k][j];
	}

	int	jmax;
	double	max = 0.0;
	for (int j = i; j < ncol(); j++)  // diagonal and right part (i <= j)
	{
	    T& sum = (*this)[i][j];
	    for (int k = 0; k < i; k++)
		sum -= (*this)[i][k] * (*this)[k][j];
	    const double tmp = fabs(sum) * scale[j];
	    if (tmp >= max)
	    {
		max  = tmp;
		jmax = j;
	    }
	}
	if (jmax != i)			// pivotting required ?
	{
	    for (int k = 0; k < nrow(); k++)	// swap i-th and jmax-th column
		swap((*this)[k][i], (*this)[k][jmax]);
	    swap(_index[i], _index[jmax]);	// swap column index
	    swap(scale[i], scale[jmax]);	// swap colum-wise scale factor
	    _det = -_det;
	}

	_det *= (*this)[i][i];

	if ((*this)[i][i] == 0.0)	// singular matrix ?
	    break;

	for (int j = i + 1; j < nrow(); j++)
	    (*this)[i][j] /= (*this)[i][i];
    }
}

template <class T> void
LUDecomposition<T>::substitute(Vector<T>& b) const
{
    if (b.dim() != ncol())
	throw std::invalid_argument("TU::LUDecomposition<T>::substitute: Dimension of given vector is not equal to mine!!");
    
    Vector<T>	tmp(b);
    for (int j = 0; j < b.dim(); j++)
	b[j] = tmp[_index[j]];

    for (int j = 0; j < b.dim(); j++)		// forward substitution
	for (int i = 0; i < j; i++)
	    b[j] -= b[i] * (*this)[i][j];
    for (int j = b.dim(); --j >= 0; )		// backward substitution
    {
	for (int i = b.dim(); --i > j; )
	    b[j] -= b[i] * (*this)[i][j];
	if ((*this)[j][j] == 0.0)		// singular matrix ?
	    throw std::runtime_error("TU::LUDecomposition<T>::substitute: singular matrix !!");
	b[j] /= (*this)[j][j];
    }
}

/************************************************************************
*  class Householder<T>							*
************************************************************************/
template <class T>
Householder<T>::Householder(const Matrix<T>& a, u_int d)
    :Matrix<T>(a), _d(d), _sigma(nrow())
{
    if (nrow() != ncol())
	throw std::invalid_argument("TU::Householder<T>::Householder: Given matrix must be square !!");
}

template <class T> void
Householder<T>::apply_from_left(Matrix<T>& a, int m)
{
    using namespace	std;
    
    if (a.nrow() < dim())
	throw std::invalid_argument("TU::Householder<T>::apply_from_left: # of rows of given matrix is smaller than my dimension !!");
    
    double	scale = 0.0;
    for (int i = m+_d; i < dim(); i++)
	scale += fabs(a[i][m]);
	
    if (scale != 0.0)
    {
	double	h = 0.0;
	for (int i = m+_d; i < dim(); i++)
	{
	    a[i][m] /= scale;
	    h += a[i][m] * a[i][m];
	}

	const double	s = (a[m+_d][m] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * a[m+_d][m];			// H = u^2 / 2
	a[m+_d][m]   += s;				// m-th col <== u
	    
	for (int j = m+1; j < a.ncol(); j++)
	{
	    T	p = 0.0;
	    for (int i = m+_d; i < dim(); i++)
		p += a[i][m] * a[i][j];
	    p /= h;					// p[j] (p' = u'A / H)
	    for (int i = m+_d; i < dim(); i++)
		a[i][j] -= a[i][m] * p;			// A = A - u*p'
	    a[m+_d][j] = -a[m+_d][j];
	}
	    
	for (int i = m+_d; i < dim(); i++)
	    (*this)[m][i] = scale * a[i][m];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_right(Matrix<T>& a, int m)
{
    using namespace	std;
    
    if (a.ncol() < dim())
	throw std::invalid_argument("Householder<T>::apply_from_right: # of column of given matrix is smaller than my dimension !!");
    
    double	scale = 0.0;
    for (int j = m+_d; j < dim(); j++)
	scale += fabs(a[m][j]);
	
    if (scale != 0.0)
    {
	double	h = 0.0;
	for (int j = m+_d; j < dim(); j++)
	{
	    a[m][j] /= scale;
	    h += a[m][j] * a[m][j];
	}

	const double	s = (a[m][m+_d] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * a[m][m+_d];			// H = u^2 / 2
	a[m][m+_d]   += s;				// m-th row <== u

	for (int i = m+1; i < a.nrow(); i++)
	{
	    T	p = 0.0;
	    for (int j = m+_d; j < dim(); j++)
		p += a[i][j] * a[m][j];
	    p /= h;					// p[i] (p = Au / H)
	    for (int j = m+_d; j < dim(); j++)
		a[i][j] -= p * a[m][j];			// A = A - p*u'
	    a[i][m+_d] = -a[i][m+_d];
	}
	    
	for (int j = m+_d; j < dim(); j++)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::apply_from_both(Matrix<T>& a, int m)
{
    using namespace	std;
    
    Vector<T>		u = a[m](m+_d, a.ncol()-m-_d);
    double		scale = 0.0;
    for (int j = 0; j < u.dim(); j++)
	scale += fabs(u[j]);
	
    if (scale != 0.0)
    {
	u /= scale;

	double		h = u * u;
	const double	s = (u[0] > 0.0 ? sqrt(h) : -sqrt(h));
	h	     += s * u[0];			// H = u^2 / 2
	u[0]	     += s;				// m-th row <== u

	Matrix<T>	A = a(m+_d, m+_d, a.nrow()-m-_d, a.ncol()-m-_d);
	Vector<T>	p = _sigma(m+_d, nrow()-m-_d);
	for (int i = 0; i < A.nrow(); i++)
	    p[i] = (A[i] * u) / h;			// p = Au / H

	const double	k = (u * p) / (h + h);		// K = u*p / 2H
	for (int i = 0; i < A.nrow(); i++)
	{				// m-th col of 'a' is used as 'q'
	    a[m+_d+i][m] = p[i] - k * u[i];		// q = p - Ku
	    for (int j = 0; j <= i; j++)		// A = A - uq' - qu'
		A[j][i] = (A[i][j] -= (u[i]*a[m+_d+j][m] + a[m+_d+i][m]*u[j]));
	}
	for (int j = 1; j < A.nrow(); j++)
	    A[j][0] = A[0][j] = -A[0][j];

	for (int j = m+_d; j < a.ncol(); j++)
	    (*this)[m][j] = scale * a[m][j];		// copy u
	_sigma[m+_d] = scale * s;
    }
}

template <class T> void
Householder<T>::make_transformation()
{
    for (int m = nrow(); --m >= 0; )
    {
	for (int i = m+1; i < nrow(); i++)
	    (*this)[i][m] = 0.0;

	if (_sigma[m] != 0.0)
	{
	    for (int i = m+1; i < nrow(); i++)
	    {
		T	g = 0.0;
		for (int j = m+1; j < ncol(); j++)
		    g += (*this)[i][j] * (*this)[m-_d][j];
		g /= (_sigma[m] * (*this)[m-_d][m]);	// g[i] (g = Uu / H)
		for (int j = m; j < ncol(); j++)
		    (*this)[i][j] -= g * (*this)[m-_d][j];	// U = U - gu'
	    }
	    for (int j = m; j < ncol(); j++)
		(*this)[m][j] = (*this)[m-_d][j] / _sigma[m];
	    (*this)[m][m] -= 1.0;
	}
	else
	{
	    for (int j = m+1; j < ncol(); j++)
		(*this)[m][j] = 0.0;
	    (*this)[m][m] = 1.0;
	}
    }
}

template <class T> bool
Householder<T>::sigma_is_zero(int m, T comp) const
{
    return (T(fabs(_sigma[m])) + comp == comp);
}

/************************************************************************
*  class QRDeomposition<T>						*
************************************************************************/
template <class T>
QRDecomposition<T>::QRDecomposition(const Matrix<T>& m)
    :Matrix<T>(m), _Qt(m.ncol(), 0)
{
    for (int j = 0; j < ncol(); ++j)
	_Qt.apply_from_right(*this, j);
    _Qt.make_transformation();
    for (int i = 0; i < nrow(); ++i)
    {
	(*this)[i][i] = _Qt.sigma()[i];
	for (int j = i + 1; j < ncol(); ++j)
	    (*this)[i][j] = 0.0;
    }
}

/************************************************************************
*  class TriDiagonal<T>							*
************************************************************************/
template <class T>
TriDiagonal<T>::TriDiagonal(const Matrix<T>& a)
    :_Ut(a, 1), _diagonal(_Ut.nrow()), _off_diagonal(_Ut.sigma())
{
    if (_Ut.nrow() != _Ut.ncol())
        throw std::invalid_argument("TU::TriDiagonal<T>::TriDiagonal: not square matrix!!");

    for (int m = 0; m < dim(); m++)
    {
	_Ut.apply_from_both(_Ut, m);
	_diagonal[m] = _Ut[m][m];
    }

    _Ut.make_transformation();
}

template <class T> void
TriDiagonal<T>::diagonalize()
{
    for (int n = dim(); --n >= 0; )
    {
	int	niter = 0;
	
#ifdef TUVectorPP_DEBUG
	std::cerr << "******** n = " << n << " ********" << std::endl;
#endif
	while (!off_diagonal_is_zero(n))
	{					// n > 0 here
	    if (niter++ > NITER_MAX)
		throw std::runtime_error("TU::TriDiagonal::diagonalize(): Number of iteration exceeded maximum value!!");

	  /* Find first m (< n) whose off-diagonal element is 0 */
	    int	m = n;
	    while (!off_diagonal_is_zero(--m));	// 0 <= m < n < dim() here

	  /* Set x and y which determine initial(i = m+1) plane rotation */
	    double	x, y;
	    initialize_rotation(m, n, x, y);
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (int i = m; ++i <= n; )
	    {
		Rotation	rot(i-1, i, x, y);
		
		_Ut.rotate_from_left(rot);

		if (i > m+1)
		    _off_diagonal[i-1] = rot.length();
		const double w = _diagonal[i] - _diagonal[i-1];
		const double d = rot.sin()*(rot.sin()*w
			       + 2.0*rot.cos()*_off_diagonal[i]);
		_diagonal[i-1]	 += d;
		_diagonal[i]	 -= d;
		_off_diagonal[i] += rot.sin()*(rot.cos()*w
				  - 2.0*rot.sin()*_off_diagonal[i]);
		if (i < n)
		{
		    x = _off_diagonal[i];
		    y = rot.sin()*_off_diagonal[i+1];
		    _off_diagonal[i+1] *= rot.cos();
		}
	    }
#ifdef TUVectorPP_DEBUG
	    std::cerr << "  niter = " << niter << ": " << off_diagonal();
#endif	    
	}
    }

    for (int m = 0; m < dim(); m++)	// sort eigen values and eigen vectors
	for (int n = m+1; n < dim(); n++)
	    if (fabs(_diagonal[n]) > fabs(_diagonal[m]))
	    {
		swap(_diagonal[m], _diagonal[n]);
		for (int j = 0; j < dim(); j++)
		{
		    const T	tmp = _Ut[m][j];
		    _Ut[m][j] = _Ut[n][j];
		    _Ut[n][j] = -tmp;
		}
	    }
}

template <class T> bool
TriDiagonal<T>::off_diagonal_is_zero(int n) const
{
    return (n == 0 || _Ut.sigma_is_zero(n, fabs(_diagonal[n-1]) +
					   fabs(_diagonal[n])));
}

template <class T> void
TriDiagonal<T>::initialize_rotation(int m, int n, double& x, double& y) const
{
    using namespace	std;
    
    const double	g = (_diagonal[n] - _diagonal[n-1]) /
			    (2.0*_off_diagonal[n]),
			absg = fabs(g),
			gg1 = (absg > 1.0 ?
			       absg * sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
			       sqrt(1.0 + absg*absg)),
			t = (g > 0.0 ? g + gg1 : g - gg1);
    x = _diagonal[m] - _diagonal[n] - _off_diagonal[n]/t;
  //x = _diagonal[m];					// without shifting
    y = _off_diagonal[m+1];
}

/************************************************************************
*  class BiDiagonal<T>							*
************************************************************************/
template <class T>
BiDiagonal<T>::BiDiagonal(const Matrix<T>& a)
    :_Dt((a.nrow() < a.ncol() ? a.ncol() : a.nrow()), 0),
     _Et((a.nrow() < a.ncol() ? a.nrow() : a.ncol()), 1),
     _diagonal(_Dt.sigma()), _off_diagonal(_Et.sigma()), _anorm(0),
     _Ut(a.nrow() < a.ncol() ? _Dt : _Et),
     _Vt(a.nrow() < a.ncol() ? _Et : _Dt)
{
    if (nrow() < ncol())
	for (int i = 0; i < nrow(); ++i)
	    for (int j = 0; j < ncol(); ++j)
		_Dt[i][j] = a[i][j];
    else
	for (int i = 0; i < nrow(); ++i)
	    for (int j = 0; j < ncol(); ++j)
		_Dt[j][i] = a[i][j];

  /* Householder reduction to bi-diagonal (off-diagonal in lower part) form */
    for (int m = 0; m < _Et.dim(); ++m)
    {
	_Dt.apply_from_right(_Dt, m);
	_Et.apply_from_left(_Dt, m);
    }

    _Dt.make_transformation();	// Accumulate right-hand transformation: V
    _Et.make_transformation();	// Accumulate left-hand transformation: U

    for (int m = 0; m < _Et.dim(); ++m)
    {
	double	anorm = fabs(_diagonal[m]) + fabs(_off_diagonal[m]);
	if (anorm > _anorm)
	    _anorm = anorm;
    }
}

template <class T> void
BiDiagonal<T>::diagonalize()
{
    for (int n = _Et.dim(); --n >= 0; )
    {
	int	niter = 0;
	
#ifdef TUVectorPP_DEBUG
	std::cerr << "******** n = " << n << " ********" << std::endl;
#endif
	while (!off_diagonal_is_zero(n))	// n > 0 here
	{
	    if (niter++ > NITER_MAX)
		throw std::runtime_error("TU::BiDiagonal::diagonalize(): Number of iteration exceeded maximum value");
	    
	  /* Find first m (< n) whose off-diagonal element is 0 */
	    int m = n;
	    do
	    {
		if (diagonal_is_zero(m-1))
		{ // If _diagonal[m-1] is zero, make _off_diagonal[m] zero.
		    double	x = _diagonal[m], y = _off_diagonal[m];
		    _off_diagonal[m] = 0.0;
		    for (int i = m; i <= n; ++i)
		    {
			Rotation	rotD(m-1, i, x, -y);

			_Dt.rotate_from_left(rotD);
			
			_diagonal[i] = -y*rotD.sin()
				     + _diagonal[i]*rotD.cos();
			if (i < n)
			{
			    x = _diagonal[i+1];
			    y = _off_diagonal[i+1]*rotD.sin();
			    _off_diagonal[i+1] *= rotD.cos();
			}
		    }
		    break;	// if _diagonal[n-1] is zero, m == n here.
		}
	    } while (!off_diagonal_is_zero(--m)); // 0 <= m < n < nrow() here.
	    if (m == n)
		break;		// _off_diagonal[n] has been made 0. Retry!

	  /* Set x and y which determine initial(i = m+1) plane rotation */
	    double	x, y;
	    initialize_rotation(m, n, x, y);
#ifdef TUBiDiagonal_DEBUG
	    std::cerr << "--- m = " << m << ", n = " << n << "---"
		      << std::endl;
	    std::cerr << "  diagonal:     " << diagonal();
	    std::cerr << "  off-diagonal: " << off_diagonal();
#endif
	  /* Apply rotation P(i-1, i) for each i (i = m+1, n+2, ... , n) */
	    for (int i = m; ++i <= n; )
	    {
	      /* Apply rotation from left */
		Rotation	rotE(i-1, i, x, y);
		
		_Et.rotate_from_left(rotE);

		if (i > m+1)
		    _off_diagonal[i-1] = rotE.length();
		T	tmp = _diagonal[i-1];
		_diagonal[i-1]	 =  rotE.cos()*tmp
				 +  rotE.sin()*_off_diagonal[i];
		_off_diagonal[i] = -rotE.sin()*tmp
				 +  rotE.cos()*_off_diagonal[i];
		if (diagonal_is_zero(i))
		    break;		// No more Given's rotation needed.
		y		 =  rotE.sin()*_diagonal[i];
		_diagonal[i]	*=  rotE.cos();

		x = _diagonal[i-1];
		
	      /* Apply rotation from right to recover bi-diagonality */
		Rotation	rotD(i-1, i, x, y);

		_Dt.rotate_from_left(rotD);

		_diagonal[i-1] = rotD.length();
		tmp = _off_diagonal[i];
		_off_diagonal[i] =  tmp*rotD.cos() + _diagonal[i]*rotD.sin();
		_diagonal[i]	 = -tmp*rotD.sin() + _diagonal[i]*rotD.cos();
		if (i < n)
		{
		    if (off_diagonal_is_zero(i+1))
			break;		// No more Given's rotation needed.
		    y		        = _off_diagonal[i+1]*rotD.sin();
		    _off_diagonal[i+1] *= rotD.cos();

		    x		        = _off_diagonal[i];
		}
	    }
#ifdef TUVectorPP_DEBUG
	    std::cerr << "  niter = " << niter << ": " << off_diagonal();
#endif
	}
    }

    for (int m = 0; m < _Et.dim(); m++)	// sort singular values and vectors
	for (int n = m+1; n < _Et.dim(); n++)
	    if (fabs(_diagonal[n]) > fabs(_diagonal[m]))
	    {
		swap(_diagonal[m], _diagonal[n]);
		for (int j = 0; j < _Et.dim(); j++)
		{
		    const T	tmp = _Et[m][j];
		    _Et[m][j] = _Et[n][j];
		    _Et[n][j] = -tmp;
		}
		for (int j = 0; j < _Dt.dim(); j++)
		{
		    const T	tmp = _Dt[m][j];
		    _Dt[m][j] = _Dt[n][j];
		    _Dt[n][j] = -tmp;
		}
	    }

    int l = _Et.dim() - 1;		// last index
    for (int m = 0; m < l; m++)		// ensure positivity of all singular
	if (_diagonal[m] < 0.0)		// values except for the last one.
	{
	    _diagonal[m] = -_diagonal[m];
	    _diagonal[l] = -_diagonal[l];
	    for (int j = 0; j < _Et.dim(); j++)
	    {
		_Et[m][j] = -_Et[m][j];
		_Et[l][j] = -_Et[l][j];
	    }
	}
}

template <class T> bool
BiDiagonal<T>::diagonal_is_zero(int n) const
{
    return _Dt.sigma_is_zero(n, _anorm);
}

template <class T> bool
BiDiagonal<T>::off_diagonal_is_zero(int n) const
{
    return _Et.sigma_is_zero(n, _anorm);
}

template <class T> void
BiDiagonal<T>::initialize_rotation(int m, int n, double& x, double& y) const
{
    using namespace	std;
    
    const double	g = ((_diagonal[n]     + _diagonal[n-1])*
			     (_diagonal[n]     - _diagonal[n-1])+
			     (_off_diagonal[n] + _off_diagonal[n-1])*
			     (_off_diagonal[n] - _off_diagonal[n-1]))
			  / (2.0*_diagonal[n-1]*_off_diagonal[n]),
      // Caution!! You have to ensure that _diagonal[n-1] != 0
      // as well as _off_diagonal[n].
			absg = fabs(g),
			gg1 = (absg > 1.0 ?
			       absg * sqrt(1.0 + (1.0/absg)*(1.0/absg)) :
			       sqrt(1.0 + absg*absg)),
			t = (g > 0.0 ? g + gg1 : g - gg1);
    x = ((_diagonal[m] + _diagonal[n])*(_diagonal[m] - _diagonal[n]) -
	 _off_diagonal[n]*(_off_diagonal[n] + _diagonal[n-1]/t)) / _diagonal[m];
  //x = _diagonal[m];				// without shifting
    y = _off_diagonal[m+1];
}

/************************************************************************
*  class Minimization1<S>						*
************************************************************************/
/*
 *  Minimize 1-dimensional function using golden section search and minima
 *  is returned in x. Minimum value of the func is also returned as a return
 *  value.
 */
template <class S> S
Minimization1<S>::minimize(S& x, S w) const
{
#define W	0.38197
    using namespace	std;

    S	x1 = x, x2 = x + w, f1 = (*this)(x1), f2 = (*this)(x2);
    
    if (f1 < f2)			// guarantee that f1 >= f2
    {
	S	tmp = x1;		// swap x1 & x2
	x1  = x2;
	x2  = tmp;
	tmp = f1;			// swap f1 & f2
	f1  = f2;
	f2  = tmp;
    }
    S	x0;
    do
    {
	x0  = x1;
	x1  = x2;
	x2 += (1.0 / W - 1.0) * (x1 - x0);	// elongate to right
#ifdef MIN1_DEBUG
	S	f0 = f1;
#endif
	f1  = f2;
	f2  = (*this)(x2);
#ifdef MIN1_DEBUG
	std::cerr << "Bracketting: [" << x0 << ", " << x1 << ", " << x2
		  << "], (" << f0 << ", " << f1 << ", " << f2 << ")"
		  << std::endl;
#endif
    } while (f1 > f2);
    
  /* Golden section search */
    S	x3 = x2;
    if (fabs(x1 - x0) > fabs(x2 - x1))
    {
	x2  = x1;
	x1 -= W * (x2 - x0);		// insert new x1 between x0 & x2
	f2  = f1;
	f1  = (*this)(x1);
    }
    else
    {
	x2 -= (1.0 - W) * (x3 - x1);	// insert new x2 between x1 & x3
	f2  = (*this)(x2);
    }
#ifdef MIN1_DEBUG
    std::cerr << "Initial:     [" << x0 << ", " << x1 << ", " << x2
	      << ", " << x3 << "], (" << f1 << ", " << f2 << ")" << std::endl;
#endif
    int	i;
    for (i = 0;
	 i < _niter_max && fabs(x3 - x0) > _tol * (fabs(x1) + fabs(x2)); ++i)
    {
	if (f1 < f2)
	{
	    x3  = x2;			// shift x2 & x3 to left
	    x2  = x1;
	    x1 -= W * (x2 - x0);	// insert new x1 between x0 & x2
	    f2  = f1;
	    f1  = (*this)(x1);
	}
	else
	{
	    x0  = x1;			// shift x0 & x1 to right
	    x1  = x2;
	    x2 += W * (x3 - x1);	// insert new x2 between x1 & x3
	    f1  = f2;
	    f2  = (*this)(x2);
	}
#ifdef MIN1_DEBUG
	std::cerr << "Golden:      [" << x0 << ", " << x1 << ", " << x2
		  << ", " << x3 << "], (" << f1 << ", " << f2 << ")"
		  << std::endl;
#endif
    }
    if (i == _niter_max)
	throw std::runtime_error("TU::Minimization1<S>::minimize(): Too many iterations!!");

    if (f1 < f2)
    {
	x = x1;
	return f1;
    }
    else
    {
	x = x2;
	return f2;
    }
}

/************************************************************************
*  class Minimization<S, T>						*
************************************************************************/
/*
 *  Minimize multi-dimensional function using conjugate gradient method and
 *  minima is returned in x. Minimum value of the func is also returned as
 *  a return value.
 */
template <class S, class T> S
Minimization<S, T>::minimize(T& x)
{
    S		val = (*this)(x);
    Vector<S>	g   = ngrad(x), h = g;
    
    for (int i = 0; i < _niter_max; ++i)
    {
	if (_print)
	    print(i, val, x);

	const S		g_sqr = g * g;
	if (g_sqr == 0.0)
	    return val;

	const S		val_next = line_minimize(x, h);
	if (near_enough(val, val_next))
	    return val_next;
	val = val_next;

	const Vector<S>	g_next = ngrad(x);
	h = g_next + (((g_next - g) * g_next) / g_sqr) * h;
	g = g_next;
	update(x);
    }

    std::cerr << "TU::Minimization<S, T>::minimize(): Too many iterations!!"
	      << std::endl;
    return val;
}

/*
 *  Minimize multi-dimensional function using steepest descent method and
 *  minima is returned in x. Minimum value of the func is also returned as
 *  a return value.
 */
template <class S, class T> S
Minimization<S, T>::steepest_descent(T& x)
{
    S		val = (*this)(x);
    Vector<S>	g   = ngrad(x);
    
    for (int i = 0; i < _niter_max; ++i)
    {
	if (_print)
	    print(i, val, x);
	
	const S		g_sqr = g * g;
	if (g_sqr == 0.0)
	    return val;

	const S		val_next = line_minimize(x, g);
	if (near_enough(val, val_next))
	    return val_next;
	val = val_next;

	g = ngrad(x);
	update(x);
    }

    std::cerr << "TU::Minimization<S, T>::steepest_descent(): Too many iterations!!"
	      << std::endl;
    return val;
}

/*
 *  Minimize function along direction h and minima is returned in x.
 *  Minimum value of the function is also returned as a return value.
 */
template <class S, class T> S
Minimization<S, T>::line_minimize(T &x, const Vector<S>& h) const
{
    LineFunction	lfunc(*this, x, h, _tol, _niter_max);
    S			d = 0.0, val = lfunc.minimize(d, 1.0);
    x = proceed(x, d * h);
    return val;
}

/*
 *  Update the status of the function to be minimized.
 */
template <class S, class T> void
Minimization<S, T>::update(const T&)
{
}

/*
 *  Print intermediate values
 */
template <class S, class T> void
Minimization<S, T>::print(int i, S val, const T& x) const
{
    std::cerr << std::setw(3) << i << ": (" << val << ')' << x;
}
 
}
