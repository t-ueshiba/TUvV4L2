/*
 *  平成19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  同所が著作権を所有する秘密情報です．著作者による許可なしにこのプロ
 *  グラムを第三者へ開示，複製，改変，使用する等の著作権を侵害する行為
 *  を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても、著作者は責任
 *  を負いません。 
 *
 *  Copyright 2007
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Author: Toshio UESHIBA
 *
 *  Confidentail and all rights reserved.
 *  This program is confidential. Any changing, copying or giving
 *  information about the source code of any part of this software
 *  and/or documents without permission by the authors are prohibited.
 *
 *  No Warranty.
 *  Authors are not responsible for any damages in the use of this program.
 *  
 *  $Id: BlockMatrix++.inst.cc,v 1.6 2007-11-26 07:28:09 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/BlockMatrix++.cc"

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
