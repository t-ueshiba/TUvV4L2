/*
 *  $Id: XilXglDC.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/XilXglDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class XilXglDC							*
************************************************************************/
/*
 *  Public member functions
 */
XilXglDC::XilXglDC(CanvasPane& parentCanvasPane, u_int w, u_int h)
    :CanvasPaneDC(parentCanvasPane, w, h),
     XilDC(parentCanvasPane, w, h),
     XglDC(parentCanvasPane, w, h)
{
}

DC&
XilXglDC::setSize(u_int width, u_int height,	u_int mul, u_int div)
{
    XilDC::setSize(width, height, mul, div);
    XglDC::setSize(width, height, mul, div);
    return *this;
}

/*
 *  Protected member functions
 */
void
XilXglDC::initializeGraphics()
{
    XilDC::initializeGraphics();
    XglDC::initializeGraphics();
}
 
}
}
