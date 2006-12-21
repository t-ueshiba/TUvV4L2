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
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/Array++.cc"
#include "TU/Vector++.cc"

namespace TU
{
template class Array<Vector<float> >;
template class Array<Vector<double> >;
template class Array2<Vector<float> >;
template class Array2<Vector<double> >;
template class Vector<float>;
template class Vector<double>;
template class Matrix<float>;
template class Matrix<double>;
template class LUDecomposition<float>;
template class LUDecomposition<double>;
template class QRDecomposition<float>;
template class QRDecomposition<double>;
template class TriDiagonal<float>;
template class TriDiagonal<double>;
template class BiDiagonal<float>;
template class BiDiagonal<double>;

template std::ostream&	operator <<(std::ostream&,
				    const Array<Vector<float> >&);
template std::ostream&	operator <<(std::ostream&,
				    const Array<Vector<double> >&);
template double		operator *(const Vector<float>&,
				   const Vector<float>&);
template double		operator *(const Vector<double>&,
				   const Vector<double>&);
template Vector<float>	operator *(const Vector<float>&,
				   const Matrix<float>&);
template Vector<double>	operator *(const Vector<double>&,
				   const Matrix<double>&);
template Matrix<float>	operator %(const Vector<float>&,
				   const Vector<float>&);
template Matrix<double>	operator %(const Vector<double>&,
				   const Vector<double>&);
template Matrix<float>	operator ^(const Vector<float>&,
				   const Matrix<float>&);
template Matrix<double>	operator ^(const Vector<double>&,
				   const Matrix<double>&);
template Matrix<float>	operator *(const Matrix<float>&,
				   const Matrix<float>&);
template Matrix<double>	operator *(const Matrix<double>&,
				   const Matrix<double>&);
template Vector<float>	operator *(const Matrix<float>&,
				   const Vector<float>&);
template Vector<double>	operator *(const Matrix<double>&,
				   const Vector<double>&);
}

#endif
