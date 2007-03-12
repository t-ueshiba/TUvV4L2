/*
 *  $Id: TUTools++.sa.cc,v 1.6 2007-03-12 07:43:51 ueshiba Exp $
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

