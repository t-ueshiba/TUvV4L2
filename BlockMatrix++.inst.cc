/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: BlockMatrix++.inst.cc,v 1.9 2008-09-10 05:10:31 ueshiba Exp $
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
