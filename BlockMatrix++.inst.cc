/*
 *  $Id: BlockMatrix++.inst.cc,v 1.3 2002-12-18 05:46:12 ueshiba Exp $
 */
#if defined __GNUG__ || defined __INTEL_COMPILER

#include "TU/BlockMatrix++.cc"
#include "TU/Array++.cc"

namespace TU
{
template class Array<Matrix<float> >;
template class Array<Matrix<double> >;
template class BlockMatrix<float>;
template class BlockMatrix<double>;

template BlockMatrix<float>
operator *(const BlockMatrix<float>&, const BlockMatrix<float>&);

template BlockMatrix<double>
operator *(const BlockMatrix<double>&, const BlockMatrix<double>&);

template Matrix<float>
operator *(const BlockMatrix<float>&, const Matrix<float>&);

template Matrix<double>
operator *(const BlockMatrix<double>&, const Matrix<double>&);

template Matrix<float>
operator *(const Matrix<float>&, const BlockMatrix<float>&);

template Matrix<double>
operator *(const Matrix<double>&, const BlockMatrix<double>&);

template Vector<float>
operator *(const BlockMatrix<float>&, const Vector<float>& v);

template Vector<double>
operator *(const BlockMatrix<double>&, const Vector<double>& v);

template Vector<float>
operator *(const Vector<float>&, const BlockMatrix<float>&);

template Vector<double>
operator *(const Vector<double>&, const BlockMatrix<double>&);
}
    
#endif
