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
 *  $Id: Image++.inst.cc,v 1.6 2003-07-06 23:53:21 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/Array++.cc"
#include "TU/Image++.cc"

namespace TU
{
template class Array<ImageLine<u_char> >;
template class Array<ImageLine<short> >;
template class Array<ImageLine<float> >;
template class Array<ImageLine<double> >;

template class Array2<ImageLine<u_char> >;
template class Array2<ImageLine<short> >;
template class Array2<ImageLine<float> >;
template class Array2<ImageLine<double> >;

template class Image<u_char>;
template class Image<short>;
template class Image<float>;
template class Image<double>;
template class Image<RGB>;
template class Image<BGR>;
template class Image<RGBA>;
template class Image<ABGR>;
template class Image<YUV444>;
template class Image<YUV422>;
template class Image<YUV411>;
}

#endif
