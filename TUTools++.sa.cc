/*
 *  $Id: TUTools++.sa.cc,v 1.4 2004-04-28 02:28:28 ueshiba Exp $
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
#if (!defined(__GNUC__) || (__GNUC__ < 3))
IOManip<Serial>	nl2cr	  (&Serial::i_nl2cr, &Serial::o_nl2crnl);
IOManip<Serial>	cr2nl	  (&Serial::i_cr2nl, &Serial::o_cr2nl);
IOManip<Serial>	upperlower(&Serial::i_upper2lower, &Serial::o_lower2upper);
IOManip<Serial>	through	  (&Serial::i_through, &Serial::o_through);
#endif
}

