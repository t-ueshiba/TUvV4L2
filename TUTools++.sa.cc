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
 *  $Id: TUTools++.sa.cc,v 1.7 2007-11-26 07:28:09 ueshiba Exp $
 */
#include "TU/Image++.h"
#include "TU/Serial++.h"

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

