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
 *  $Id: Array++.cc,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#include "TU/Array++.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Array<T>							*
************************************************************************/
inline u_int
_pdim(u_int i, u_int d, u_int a)
{
    return (i+d <= a ? d : i < a ? a-i : 0);
}

template <class T>
Array<T>::Array(const Array<T>& a, u_int i, u_int d)
    :_d(_pdim(i, d, a.dim())),
     _p(dim() > 0 ? (T*)&a[i] : 0), _siz(dim()), _shr(1)
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array<T>::Array(const Array<T>&, u_int, u_int) invoked!\n  this = "
	      << (void*)this << ", dim = " << dim()
	      << std::endl;
#endif
}

template <class T>
Array<T>::Array(const Array<T>& a)
    :_d(a.dim()), _p(new T[dim()]), _siz(dim()), _shr(0)
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array<T>::Array(const Array<T>&) invoked!\n"
	      << "  this = " << (void*)this << ", dim = " << dim()
	      << std::endl;		
#endif
    for (int i = 0; i < dim(); i++)
	(*this)[i] = a[i];
}

template <class T> Array<T>&
Array<T>::operator =(const Array<T>& a)
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array<T>::operator =(const Array<T>&) invoked!"
	      << std::endl;
#endif
    if (_p != a._p)
    {
	resize(a.dim());
	for (int i = 0; i < dim(); i++)
	    (*this)[i] = a[i];
    }
    return *this;
}

template <class T> const T&
Array<T>::at(int i) const
{
    if (i < 0 || i >= dim())
	throw std::out_of_range("TU::Array<T>::at: invalid index!");
    return _p[i];
}

template <class T> T&
Array<T>::at(int i)
{
    if (i < 0 || i >= dim())
	throw std::out_of_range("TU::Array<T>::at: invalid index!");
    return _p[i];
}

template <class T> Array<T>&
Array<T>::operator =(double c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] = c;
    return *this;
}

template <class T> Array<T>&
Array<T>::operator *=(double c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] *= c;
    return *this;
}
    
template <class T> Array<T>&
Array<T>::operator /=(double c)
{
    for (int i = 0; i < dim(); )
	(*this)[i++] /= c;
    return *this;
}

template <class T> Array<T>&
Array<T>::operator +=(const Array<T>& a)
{
    check_dim(a.dim());
    for (int i = 0; i < dim(); i++)
	(*this)[i] += a[i];
    return *this;
}

template <class T> Array<T>&
Array<T>::operator -=(const Array<T>& a)
{
    check_dim(a.dim());
    for (int i = 0; i < dim(); i++)
	(*this)[i] -= a[i];
    return *this;
}

template <class T> bool
Array<T>::operator ==(const Array<T>& a) const
{
    if (dim() != a.dim())
	return false;
    for (int i = 0; i < dim(); ++i)
	if ((*this)[i] != a[i])
	    return false;
    return true;
}

template <class T> std::istream&
Array<T>::restore(std::istream& in)
{
    in.read((char*)_p, sizeof(T) * dim());
    return in;
}

template <class T> std::ostream&
Array<T>::save(std::ostream& out) const
{
    out.write((const char*)(T*)(*this), sizeof(T) * dim());
    return out;
}

template <class T> bool
Array<T>::resize(u_int d)
{
    if (dim() == d)
	return false;
    
    if (_shr)
	throw std::logic_error("Array<T>::resize: cannot change dimension of shared array!");

    const u_int	old_dim = dim();
    if (_siz < (_d = d))
    {
	delete [] _p;
	_siz = dim();
	_p = new T[_siz];
    }
    return dim() > old_dim;
}

template <class T> void
Array<T>::resize(T* p, u_int d)
{
    _d = d;
    if (!_shr)
	delete [] _p;
    _p = p;
    _shr = 1;
    _siz = d;
}

template <class T> void
Array<T>::check_dim(u_int d) const
{
    if (dim() != d)
	throw
	    std::invalid_argument("Array<T>::check_dim: dimension mismatch!");
}

template <class T> std::istream&
Array<T>::get(std::istream& in, int m)
{
#define TUArrayPPBufSiz	2048
    Array<T>	tmp(TUArrayPPBufSiz);
    int		n = 0;
    
    while (n < tmp.dim())
    {
	char	c;
	while (in.get(c))		// skip white spaces other than '\n'
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(m + n);
	    break;
	}

	in.putback(c);
	in >> tmp[n++];
    }
    if (n == tmp.dim())
	get(in, m + n);

    for (int i = 0; i < n; ++i)
	(*this)[m + i] = tmp[i];

    return in;
}

template <class T> std::ostream&
operator <<(std::ostream& out, const Array<T>& a)
{
    for (int i = 0; i < a.dim(); )
	out << ' ' << a[i++];
    return out << std::endl;
}

/************************************************************************
*  class Array2<T>							*
************************************************************************/
template <class T>
Array2<T>::Array2(const Array2<T>& a, u_int i, u_int j, u_int r, u_int c)
    :Array<T>(_pdim(i, r, a.nrow())), _cols(_pdim(j, c, a.ncol())),
     _ary((nrow() > 0 && ncol() > 0 ? (ET*)&a[i][j] : 0), nrow()*ncol())
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array2<T>::Array2(const Array2<T>&, u_int, u_int, u_int, u_int) invoked!\n"
	      << "  this = " << (void*)this << ", " << nrow() << 'x' << ncol()
	      << std::endl;
#endif
    for (int ii = 0; ii < nrow(); ii++)
	(*this)[ii].resize((ET*)&a[i+ii][j], ncol());
}    

template <class T>
Array2<T>::Array2(const Array2<T>& a)
    :Array<T>(a.nrow()), _cols(a.ncol()), _ary(nrow()*ncol())
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array2<T>::Array2(const Array2<T>&) invoked!\n"
	      << "  this = " << (void*)this << ", " << nrow() << 'x' << ncol()
	      << std::endl;
#endif
    set_rows();
    for (int i = 0; i < nrow(); i++)
	(*this)[i] = a[i];
}    

template <class T>
Array2<T>::~Array2()
{
}

template <class T> Array2<T>&
Array2<T>::operator =(const Array2<T>& a)
{
#ifdef TUArrayPP_DEBUG
    std::cerr << "TU::Array2<T>::operator =(const Array2<T>&) invoked!"
	      << std::endl;
#endif
    resize(a.nrow(), a.ncol());
    for (int i = 0; i < nrow(); i++)
	(*this)[i] = a[i];
    return *this;
}

template <class T> std::istream&
Array2<T>::restore(std::istream& in)
{
    for (int i = 0; i < nrow(); )
	(*this)[i++].restore(in);
    return in;
}

template <class T> std::ostream&
Array2<T>::save(std::ostream& out) const
{
    for (int i = 0; i < nrow(); )
	(*this)[i++].save(out);
    return out;
}

template <class T> void
Array2<T>::resize(u_int r, u_int c)
{
    const u_int	old_ncol = ncol();

    _cols = c;
    _ary.resize(r*ncol());
    if (Array<T>::resize(r) || ncol() != old_ncol)
	set_rows();
}

template <class T> void
Array2<T>::resize(ET* p, u_int r, u_int c)
{
    _cols = c;
    _ary.resize(p, r*ncol());
    Array<T>::resize(r);
    set_rows();
}

template <class T> void
Array2<T>::set_rows()
{
    for (int i = 0; i < nrow(); i++)
	(*this)[i].resize((ET*)*this + i*ncol(), ncol());
}

template <class T> std::istream&
Array2<T>::get(std::istream& in, int i, int j, int jmax)
{
    char		c;

    while (in.get(c))			// Skip white spaces other than '\n'.
	if (!isspace(c) || c == '\n')
	    break;

    if (!in || c == '\n')
    {
	++i;				// Proceed to the next row.
	if (j > jmax)
	    jmax = j;
	j = 0;				// Return to the first column.

	while (in.get(c))		// Skip white spaces other than '\n'.
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(i, jmax);
	    return in;
	}
    }
    in.putback(c);
    ET val;
    in >> val;
    get(in, i, j + 1, jmax);
    (*this)[i][j] = val;
    return in;
}    
 
}
