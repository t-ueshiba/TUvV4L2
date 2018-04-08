/*
 *  $Id$  
 */
#ifndef TU_V_CANVASPANEDC_H
#define TU_V_CANVASPANEDC_H

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
		 size_t width=0, size_t height=0, float zoom=1)		;
    virtual		~CanvasPaneDC()					;
    
    virtual const Widget&	widget()			const	;

    virtual DC&		setSize(size_t width, size_t height, float zoom);
    virtual void	callback(CmdId id, CmdVal val)			;
	    void	grabKeyboard()				const	;

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
    virtual size_t	realWidth()				const	;
    virtual size_t	realHeight()				const	;

    const Widget	_widget;		// vCanvasWidget
    Menu		_popup;
    int			_u_last;
    int			_v_last;
};

}
}
#endif	// !TU_V_CANVASPANEDC_H
