/*
 *  $Id: CanvasPaneDC.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvCanvasPaneDC_h
#define __TUvCanvasPaneDC_h

#include "TU/v/XDC.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/Menu.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CanvasPaneDC							*
************************************************************************/
class CanvasPaneDC : public Object, public XDC
{
  public:
    CanvasPaneDC(CanvasPane& parentCanvasPane,
		 u_int width=0, u_int height=0)				;
    virtual		~CanvasPaneDC()					;
    
    virtual const Widget&	widget()			const	;

    virtual DC&		setSize(u_int width, u_int height,
				u_int mul,   u_int div)			;
    virtual void	callback(CmdId id, CmdVal val)			;

  protected:
    virtual Drawable	drawable()				const	;
    virtual void	initializeGraphics()				;
    virtual DC&		repaintUnderlay()				;
    virtual DC&		repaintOverlay()				;

  private:
    friend void		EVcanvasPaneDC(::Widget,
				       XtPointer client_data,
				       XEvent* event,
				       Boolean*)			;
    friend void		CBcanvasPaneDC(::Widget,
				       XtPointer client_data,
				       XtPointer)			;

    void		setDeviceSize()					;
    virtual u_int	realWidth()				const	;
    virtual u_int	realHeight()				const	;

    const Widget	_widget;		// vCanvasWidget
    Menu		_popup;
    int			_u_last;
    int			_v_last;
};

}
}
#endif	// !__TUvCanvasPaneDC_h
