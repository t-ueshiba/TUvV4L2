/*
 *  $Id$  
 */
#ifndef TU_V_CANVASPANE_H
#define TU_V_CANVASPANE_H

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
    CanvasPane(Window& parentWin, size_t devWidth=0, size_t devHeight=0);
    virtual			~CanvasPane()				;

    virtual const Widget&	widget()			const	;

    virtual void		repaintUnderlay()			;
    virtual void		repaintOverlay()			;
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
#endif	// !TU_V_CANVASPANE_H
