/*
 *  $Id: App.h,v 1.2 2002-07-25 02:38:09 ueshiba Exp $
 */
#ifndef __TUvApp_h
#define __TUvApp_h

#include "TU/v/TUv++.h"
#include "TU/v/Colormap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class App								*
************************************************************************/
class App : public Window
{
  public:
    App(int& argc, char* argv[])					;
    virtual ~App()							;

    virtual const Widget&	widget()			const	;
    virtual Colormap&		colormap()				;
    virtual void		callback(CmdId id, CmdVal val)		;

	    void	run()						;
    virtual void	exit()						;

	    void	addColormapWindow(const Window& vWindow)const;
    
  protected:
    virtual App&	app()						;

  private:
    XtAppContext	_appContext;
    const Widget	_widget;		// applicationShellWidget
    Colormap		_colormap;
    bool		_active;
};

}
}
#endif	// !__TUvApp_h
