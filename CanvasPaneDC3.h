/*
 *  $Id: CanvasPaneDC3.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvCanvasPaneDC3_h
#define __TUvCanvasPaneDC3_h

#include "TU/v/CanvasPaneDC.h"
#include "TU/v/DC3.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPaneDC3							*
************************************************************************/
class CanvasPaneDC3 : virtual public CanvasPaneDC, public DC3
{
  public:
    CanvasPaneDC3(CanvasPane& parentCanvasPane,
		  u_int width=0, u_int height=0)			;
    virtual		~CanvasPaneDC3()				;
    virtual void	callback(CmdId id, CmdVal val)			;

  protected:
    virtual void	initializeGraphics()				;
};

}
}
#endif	// !__TUvCanvasPaneDC3_h
