/*
 *  $Id$  
 */
#ifndef TU_V_CMDWINDOW_H
#define TU_V_CMDWINDOW_H

#include "TU/v/TUv++.h"
#include "TU/v/Colormap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CmdWindow							*
************************************************************************/
class CmdWindow : public Window
{
  private:
    class Paned : public Object
    {
      public:
	Paned(CmdWindow&)					;
	virtual			~Paned()			;
	
	virtual const Widget&	widget()		const	;

      private:
	const Widget	_widget;			// gridboxWidget
    };

  public:
    CmdWindow(Window&			parentWindow,
	      const char*		myName,
	      Colormap::Mode		mode,
	      size_t			resolution,
	      size_t			underlayCmapDim,
	      size_t			overlayDepth,
	      int			screen=-1,
	      bool			fullScreen=false)	;
    CmdWindow(Window&			parentWindow,
	      const char*		myName,
	      const XVisualInfo*	vinfo,
	      Colormap::Mode		mode,
	      size_t			resolution,
	      size_t			underlayCmapDim,
	      size_t			overlayDepth,
	      bool			fullScreen=false)	;
    virtual			~CmdWindow()			;

    virtual const Widget&	widget()		const	;
    virtual Colormap&		colormap()			;
    virtual void		show()				;

  protected:
    virtual Object&		paned()				;

  private:
    friend void		EVcmdWindow(::Widget widget, XtPointer cmdWindowPtr,
				    XEvent* event, Boolean*);
    
    Colormap		_colormap;
    const Widget	_widget;		// applicationShellWidget
    Paned		_paned;
};

}
}
#endif	// !TU_V_CMDWINDOW_H
