/*
 *  $Id: CmdWindow.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvCmdWindow_h
#define __TUvCmdWindow_h

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
	      u_int			resolution,
	      u_int			underlayCmapDim,
	      u_int			overlayDepth)		;
    CmdWindow(Window&			parentWindow,
	      const char*		myName,
	      const XVisualInfo*	vinfo,
	      Colormap::Mode		mode,
	      u_int			resolution,
	      u_int			underlayCmapDim,
	      u_int			overlayDepth)		;
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
#endif	// !__TUvCmdWindow_h
