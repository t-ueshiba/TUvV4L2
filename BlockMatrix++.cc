/*
 *  $Id: BlockMatrix++.cc,v 1.4 2007-02-28 00:16:06 ueshiba Exp $
 */
#include "TU/BlockMatrix++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class BlockMatrix<T>							*
************************************************************************/
template <class T>
BlockMatrix<T>::BlockMatrix(const Array<u_int>& nrows,
			    const Array<u_int>& ncols)
    :Array<Matrix<T> >(nrows.dim())
{
    if (nrows.dim() != ncols.dim())
	throw std::invalid_argument("TU::BlockMatrix<T>::BlockMatrix: dimension mismatch between nrows and ncols!!");
    for (int i = 0; i < dim(); ++i)
    {
	(*this)[i].resize(nrows[i], ncols[i]);
	(*this)[i] = 0.0;
    }
}

template <class T> u_int
BlockMatrix<T>::nrow() const
{
    size_t	r = 0;
    for (int i = 0; i < dim(); ++i)
	r += (*this)[i].nrow();
    return r;
}

template <class T> u_int
BlockMatrix<T>::ncol() const
{
    size_t	c = 0;
    for (int i = 0; i < dim(); ++i)
	c += (*this)[i].ncol();
    return c;
}

template <class T> BlockMatrix<T>
BlockMatrix<T>::trns() const
{
    BlockMatrix	val(dim());
    for (int i = 0; i < val.dim(); ++i)
	val[i] = (*this)[i].trns();
    return val;
}

template <class T> BlockMatrix<T>&
BlockMatrix<T>::operator =(T c)
{
    for (int i = 0; i < dim(); ++i)
	(*this)[i] = c;
    return *this;
}
    
template <class T>
BlockMatrix<T>::operator Matrix<T>() const
{
    Matrix<T>	val(nrow(), ncol());
    for (int r = 0, c = 0, i = 0; i < dim(); ++i)
    {
	val(r, c, (*this)[i].nrow(), (*this)[i].ncol()) = (*this)[i];
	r += (*this)[i].nrow();
	c += (*this)[i].ncol();
    }
    return val;
}

/************************************************************************
*  numeric operators							*
************************************************************************/
template <class T> BlockMatrix<T>
operator *(const BlockMatrix<T>& a, const BlockMatrix<T>& b)
{
    a.check_dim(b.dim());
    BlockMatrix<T>	val(a.dim());
    for (int i = 0; i < val.dim(); ++i)
	val[i] = a[i] * b[i];
    return val;
}

template <class T> Matrix<T>
operator *(const BlockMatrix<T>& b, const Matrix<T>& m)
{
    Matrix<T>	val(b.nrow(), m.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(r, 0, b[i].nrow(), m.ncol())
	    = b[i] * m(c, 0, b[i].ncol(), m.ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != m.nrow())
	throw std::invalid_argument("TU::operaotr *(const BlockMatrix<T>&, const Matrix<T>&): dimension mismatch!!");
    return val;
}

template <class T> Matrix<T>
operator *(const Matrix<T>& m, const BlockMatrix<T>& b)
{
    Matrix<T>	val(m.nrow(), b.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(0, c, m.nrow(), b[i].ncol())
	    = m(0, r, m.nrow(), b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != m.ncol())
	throw std::invalid_argument("TU::operaotr *(const Matrix<T>&, const BlockMatrix<T>&): dimension mismatch!!");
    return val;
}

template <class T> Vector<T>
operator *(const BlockMatrix<T>& b, const Vector<T>& v)
{
    Vector<T>	val(b.nrow());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(r, b[i].nrow()) = b[i] * v(c, b[i].ncol());
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (c != v.dim())
	throw std::invalid_argument("TU::operaotr *(const BlockMatrix<T>&, const Vector<T>&): dimension mismatch!!");
    return val;
}

template <class T> Vector<T>
operator *(const Vector<T>& v, const BlockMatrix<T>& b)
{
    Vector<T>	val(b.ncol());
    int		r = 0, c = 0;
    for (int i = 0; i < b.dim(); ++i)
    {
	val(c, b[i].ncol()) = v(r, b[i].nrow()) * b[i];
	r += b[i].nrow();
	c += b[i].ncol();
    }
    if (r != v.dim())
	throw std::invalid_argument("TU::operaotr *(const Vector<T>&, const BlockMatrix<T>&): dimension mismatch!!");
    return val;
}
 
}
