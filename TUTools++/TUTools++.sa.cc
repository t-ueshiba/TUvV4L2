/*
 *  $Id: TUTools++.sa.cc,v 1.3 2003-07-06 23:53:22 ueshiba Exp $
 */
#include "TU/Image++.h"
#ifndef __APPLE__
#  include "TU/Serial++.h"
#endif

namespace TU
{
/************************************************************************
*  Color space converter form YUVxxx					*
************************************************************************/
const ConversionFromYUV	conversionFromYUV;

/************************************************************************
*  Manipulators for Serial						*
************************************************************************/
#ifndef __APPLE__
IOManip<Serial>	nl2cr	  (&Serial::i_nl2cr, &Serial::o_nl2crnl);
IOManip<Serial>	cr2nl	  (&Serial::i_cr2nl, &Serial::o_cr2nl);
IOManip<Serial>	upperlower(&Serial::i_upper2lower, &Serial::o_lower2upper);
IOManip<Serial>	through	  (&Serial::i_through, &Serial::o_through);
#endif
}

