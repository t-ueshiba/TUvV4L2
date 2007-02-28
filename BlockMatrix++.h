/*
 *  $Id: BlockMatrix++.h,v 1.4 2007-02-28 00:16:06 ueshiba Exp $
 */
#ifndef __TUBlockMatrixPP_h
#define __TUBlockMatrixPP_h

#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class BlockMatrix<T>							*
************************************************************************/
template <class T>
class BlockMatrix : public Array<Matrix<T> >
{
  public:
    explicit BlockMatrix(u_int d=0)	:Array<Matrix<T> >(d)	{}
    BlockMatrix(const Array<u_int>& nrows,
		const Array<u_int>& ncols)			;

    using		Array<Matrix<T> >::dim;
    u_int		nrow()				const	;
    u_int		ncol()				const	;
    BlockMatrix		trns()				const	;
    BlockMatrix&	operator  =(T c)			;
    BlockMatrix&	operator *=(double c)
			{Array<Matrix<T> >::operator *=(c); return *this;}
    BlockMatrix&	operator /=(double c)
			{Array<Matrix<T> >::operator /=(c); return *this;}
    BlockMatrix&	operator +=(const BlockMatrix& b)
			{Array<Matrix<T> >::operator +=(b); return *this;}
    BlockMatrix&	operator -=(const BlockMatrix& b)
			{Array<Matrix<T> >::operator -=(b); return *this;}
			operator Matrix<T>()		const	;
};

/************************************************************************
*  numeric operators							*
************************************************************************/
template <class T> BlockMatrix<T>
operator *(const BlockMatrix<T>& a, const BlockMatrix<T>& b)	;

template <class T> Matrix<T>
operator *(const BlockMatrix<T>& b, const Matrix<T>& m)		;

template <class T> Matrix<T>
operator *(const Matrix<T>& m, const BlockMatrix<T>& b)		;

template <class T> Vector<T>
operator *(const BlockMatrix<T>& b, const Vector<T>& v)		;

template <class T> Vector<T>
operator *(const Vector<T>& v, const BlockMatrix<T>& b)		;
 
}

#endif	/* !__TUBlockMatrixPP_h	*/
