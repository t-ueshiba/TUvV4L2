/*
 *  $Id: MemoryDC.h,v 1.3 2004-09-27 08:32:00 ueshiba Exp $
 */
#ifndef __TUvMemoryDC_h
#define __TUvMemoryDC_h

#include "TU/v/XDC.h"
#include "TU/v/CanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MemoryDC							*
************************************************************************/
class MemoryDC : public XDC
{
  public:
    MemoryDC(Colormap& colormap, u_int width, u_int height)		;
    virtual		~MemoryDC()					;

    DC&			setSize(u_int width, u_int height,
				u_int mul, u_int div)			;

  protected:
    virtual Drawable	drawable()				const	;
    virtual void	initializeGraphics()				;
    virtual DC&		repaintUnderlay()				;
    virtual DC&		repaintOverlay()				;

  private:
    Pixmap	_pixmap;
};

}
}
#endif	// !__TUvMemoryDC_h
