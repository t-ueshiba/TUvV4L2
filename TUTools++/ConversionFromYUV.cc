/*
 *  $Id: ConversionFromYUV.cc,v 1.3 2007-03-06 07:15:31 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
static inline int	flt2fix(float flt)	{return int(flt * (1 << 10));}

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
