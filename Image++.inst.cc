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
 *  $Id: Image++.inst.cc,v 1.10 2007-11-26 07:28:09 ueshiba Exp $
 */
#if defined(__GNUG__) || defined(__INTEL_COMPILER)

#include "TU/Image++.cc"

namespace TU
{
template class Image<u_char>;
template class Image<short>;
template class Image<int>;
template class Image<float>;
template class Image<double>;
template class Image<RGB>;
template class Image<BGR>;
template class Image<RGBA>;
template class Image<ABGR>;
template class Image<YUV444>;
template class Image<YUV422>;
template class Image<YUV411>;

template class IntegralImage<int>;
template IntegralImage<int>&
IntegralImage<int>::initialize(const Image<u_char>& image)		;
template const IntegralImage<int>&
IntegralImage<int>::crossVal(Image<int>& out, int cropSize)	const	;
template const IntegralImage<int>&
IntegralImage<int>::crossVal(Image<float>& out, int cropSize)	const	;
    
template class IntegralImage<float>;
template IntegralImage<float>&
IntegralImage<float>::initialize(const Image<u_char>& image)		;
template IntegralImage<float>&
IntegralImage<float>::initialize(const Image<float>& image)		;
template const IntegralImage<float>&
IntegralImage<float>::crossVal(Image<float>& out, int cropSize)	const	;

template class DiagonalIntegralImage<int>;
template DiagonalIntegralImage<int>&
DiagonalIntegralImage<int>::initialize(const Image<u_char>& image)	;
template const DiagonalIntegralImage<int>&
DiagonalIntegralImage<int>::crossVal(Image<int>& out, int cropSize)	const;
template const DiagonalIntegralImage<int>&
DiagonalIntegralImage<int>::crossVal(Image<float>& out, int cropSize)	const;
    
template class DiagonalIntegralImage<float>;
template DiagonalIntegralImage<float>&
DiagonalIntegralImage<float>::initialize(const Image<u_char>& image)	;
template DiagonalIntegralImage<float>&
DiagonalIntegralImage<float>::initialize(const Image<float>& image)	;
template const DiagonalIntegralImage<float>&
DiagonalIntegralImage<float>::crossVal(Image<float>& out, int cropSize)	const;
}

#endif
