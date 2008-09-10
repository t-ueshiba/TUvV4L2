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
 *  $Id: TUTools++.sa.cc,v 1.11 2008-09-10 05:10:47 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  Color space converter form YUVxxx					*
************************************************************************/
const ConversionFromYUV	conversionFromYUV;

/************************************************************************
*  Manipulators for Serial						*
************************************************************************/
IOManip<Serial>	nl2cr	  (&Serial::i_nl2cr, &Serial::o_nl2crnl);
#ifndef __APPLE__
IOManip<Serial>	cr2nl	  (&Serial::i_cr2nl, &Serial::o_cr2nl);
IOManip<Serial>	upperlower(&Serial::i_upper2lower, &Serial::o_lower2upper);
#endif
IOManip<Serial>	through	  (&Serial::i_through, &Serial::o_through);
}

