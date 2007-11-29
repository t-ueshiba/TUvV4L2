/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Vector++.inst.cc,v 1.9 2007-11-29 07:06:37 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/Vector++.cc"

namespace TU
{
template class Vector<short,  FixedSizedBuf<short,  2> >;
template class Vector<int,    FixedSizedBuf<int,    2> >;
template class Vector<float,  FixedSizedBuf<float,  2> >;
template class Vector<double, FixedSizedBuf<double, 2> >;

template class Vector<short,  FixedSizedBuf<short,  3> >;
template class Vector<int,    FixedSizedBuf<int,    3> >;
template class Vector<float,  FixedSizedBuf<float,  3> >;
template class Vector<double, FixedSizedBuf<double, 3> >;

template class Vector<short,  FixedSizedBuf<short,  4> >;
template class Vector<int,    FixedSizedBuf<int,    4> >;
template class Vector<float,  FixedSizedBuf<float,  4> >;
template class Vector<double, FixedSizedBuf<double, 4> >;

template class Matrix<double, FixedSizedBuf<double, 9> >;

template class Vector<float>;
template class Vector<double>;

template class Matrix<float>;
template class Matrix<double>;

template class LUDecomposition<float>;
template class LUDecomposition<double>;

template class Householder<float>;
template class Householder<double>;

template class QRDecomposition<float>;
template class QRDecomposition<double>;

template class TriDiagonal<float>;
template class TriDiagonal<double>;

template class BiDiagonal<float>;
template class BiDiagonal<double>;
}

#endif
