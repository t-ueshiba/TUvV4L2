/*
 *  $Id$  
 */
#ifndef TU_V_CANVASPANEDC3_H
#define TU_V_CANVASPANEDC3_H

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
		  size_t width=0, size_t height=0, float zppm=1)	;
    virtual		~CanvasPaneDC3()				;
    virtual void	callback(CmdId id, CmdVal val)			;

  protected:
    virtual void	initializeGraphics()				;
};

}
}
#endif	// !TU_V_CANVASPANEDC3_H
