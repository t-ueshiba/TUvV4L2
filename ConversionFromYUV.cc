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
 *  $Id$
 */
#include "TU/Image++.h"

namespace TU
{
static inline int	flt2fix(float flt)	{return int(flt * (1 << 10));}

/************************************************************************
*  global variables							*
************************************************************************/
__PORT const ConversionFromYUV	conversionFromYUV;

/************************************************************************
*  class ConversionFromYUV						*
************************************************************************/
ConversionFromYUV::ConversionFromYUV()
{
    for (int i = 0; i < 256; ++i)
    {
	_r [i] = int(1.4022f * (i - 128));
	_g0[i] = flt2fix(0.7144f * (i - 128));
	_g1[i] = flt2fix(0.3457f * (i - 128));
	_b [i] = int(1.7710f * (i - 128));
    }
}
 
}
