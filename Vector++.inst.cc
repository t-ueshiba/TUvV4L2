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
 *  $Id: Vector++.inst.cc,v 1.7 2007-11-26 07:28:09 ueshiba Exp $
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
