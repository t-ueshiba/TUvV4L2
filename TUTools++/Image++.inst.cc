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
 *  $Id: Image++.inst.cc,v 1.12 2007-11-29 07:06:36 ueshiba Exp $
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
