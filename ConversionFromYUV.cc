/*
 *  $Id: ConversionFromYUV.cc,v 1.2 2002-07-25 02:38:04 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
inline int	flt2fix(float flt)	{return int(flt * (1 << 10));}

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
