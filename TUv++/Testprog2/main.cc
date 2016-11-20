/*
 *  $Id: main.cc,v 1.3 2008-05-27 11:38:27 ueshiba Exp $
 */
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/v/ShmDC.h"
#include <exception>

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_Slider};

static float	range[] = {0, 2, 0.1};
static CmdDef	Cmds[] =
{
    {C_Slider, c_Slider, 1.0f, "Zoom:", range, CA_None, 0, 0, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const Image<BGR>& image)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _dc(*this), _image(image)					{}

    CanvasPaneDC&	dc()					{return _dc;}
    
    virtual void	repaintUnderlay()			;

  private:
      //  CanvasPaneDC		_dc;
    ShmDC		_dc;
    const Image<BGR>&	_image;
};

void
MyCanvasPane::repaintUnderlay()
{
    _dc << clear << _image;
}

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&			parentApp,
		const char*		name,
		const Image<BGR>&	image)			;

    virtual void	callback(CmdId id, CmdVal val)		;
    
  private:
    CmdPane		_cmd;
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const Image<BGR>& image)
    :CmdWindow(parentApp, name, 0, Colormap::RGBColor, 16, 0, 0),
     _cmd(*this, Cmds), _canvas(*this, image)
{
    _cmd.place(0, 0, 1, 1);
    _canvas.place(0, 1, 1, 1);
    show();
}

void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_Slider:
	_canvas.dc().setZoom(val.f).repaintAll();
	break;
    }
}
 
}
}
/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    v::App		vapp(argc, argv);
    try
    {
	Image<BGR>	image;
	image.restore(cin);

	v::MyCmdWindow	myWin0(vapp, "Image viewer (0)", image);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }

    cerr << "Loop exited!" << endl;

    return 0;
}
