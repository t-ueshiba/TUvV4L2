/*
 *  $Id: XilXglDC.h,v 1.2 2002-07-25 02:38:08 ueshiba Exp $
 */
#ifndef __TUvXilXglDC_h
#define __TUvXilXglDC_h

#include "TU/v/XilDC.h"
#include "TU/v/XglDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class XilXglDC							*
************************************************************************/
class XilXglDC : public XilDC, public XglDC
{
  public:
    XilXglDC(CanvasPane& parentCanvasPane, u_int width=0, u_int height=0);

    virtual DC&	setSize(u_int width, u_int height, u_int mul, u_int div);

  protected:
    virtual void	initializeGraphics()				;
};
 
}
}
#endif	/* ! __TUvXilXglDC_h */
