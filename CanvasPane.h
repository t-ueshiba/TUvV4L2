/*
 *  $Id: CanvasPane.h,v 1.2 2002-07-25 02:38:09 ueshiba Exp $
 */
#ifndef __TUvCanvasPane_h
#define __TUvCanvasPane_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPane							*
************************************************************************/
class CanvasPane : public Pane
{
  public:
    CanvasPane(Window& parentWin, u_int devWidth=0, u_int devHeight=0)	;
    virtual			~CanvasPane()				;

    virtual const Widget&	widget()			const	;

    virtual void		repaintUnderlay(int u, int v,
						int width, int height)	;
    virtual void		repaintOverlay(int u, int v,
					       int width, int height)	;
    void			moveDC(int u, int v)			;
    
  protected:
    virtual CanvasPane&		canvasPane()				;
    virtual void		initializeGraphics()			;
    
  private:
  // allow access to initializeGraphics
    friend void		CBcanvasPaneDC(::Widget, XtPointer client_data,
				       XtPointer)			;

    const Widget	_widget;		// viewportWidget
};

}
}
#endif	// !__CanvasPane_h
