/*
 *  $Id: main.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include <iomanip>
#include <fstream>
#include <sstream>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CmdPane.h"
#include "TU/v/XilDC.h"

namespace TU
{
typedef Array2<Array<int> >	IntArray2;

namespace v
{
/************************************************************************
*  menus and commands							*
************************************************************************/
enum	{
	    c_Slider=100, c_Underlay, c_Overlay
	};

static MenuDef FileMenu[] =
{
    {"New",  M_New,  false, noSub},
    {"Open", M_Open, false, noSub},
    {"-",    M_Line, false, noSub},
    {"Quit", M_Exit, false, noSub},
    EndOfMenu
};

static CmdDef MainMenu[] =
{
  {C_MenuButton, M_File, 0, "File", FileMenu, CA_None, 0, 0, 1, 1, 0},
  EndOfCmds
};

static int	range[] = {0, 200, 100};

static CmdDef Cmds[] =
{
  {C_Slider,	   c_Slider, 100, "Zoom:",    range,  CA_None, 0, 0, 2, 1, 0},
  {C_ToggleButton, c_Underlay, 1, "Underlay", noProp, CA_None, 0, 1, 1, 1, 0},
  {C_ToggleButton, c_Overlay,  1, "Overlay",  noProp, CA_None, 1, 1, 1, 1, 0},
  EndOfCmds
};

/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
static BGR	bgr[] =
		{
		    BGR(255, 0, 0),
		    BGR(0, 255, 0),
		    BGR(0, 0, 255),
		    BGR(0, 255, 255),
		    BGR(255, 0, 255),
		    BGR(255, 255, 0)
		};

class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const XilImage<u_char>& image,
		 const IntArray2& points)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _dc(*this),
	 _image(image), _points(points), _underlay(1), _overlay(1)	{}

    CanvasPaneDC&	dc()					{return _dc;}
    
    virtual void	repaintUnderlay(int x, int y, int w, int h)	;
    virtual void	repaintOverlay(int x, int y, int w, int h)	;

  private:
    XilDC			_dc;
    const XilImage<u_char>&	_image;
    const IntArray2&		_points;

  public:
    int				_underlay, _overlay;
};

void
MyCanvasPane::repaintUnderlay(int, int, int, int)
{
    if (!_underlay)
	return;

    _dc << clear << _image;
}

void
MyCanvasPane::repaintOverlay(int, int, int, int)
{
    if (!_overlay)
	return;

    _dc << clear << cross;
    for (u_int i = 0; i < _points.dim(); ++i)
	_dc << foreground(1+i) << Point2<int>(_points[i][0], _points[i][1]);
}

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&				parentApp,
		const char*			name,
		const XilImage<u_char>&	image,
		const IntArray2&		points,
		u_int				resolution,
		u_int				underlayCmapDim,
		u_int				overlayCmapDim)		;

    virtual void	callback(CmdId, CmdVal)				;
    
  private:
    CmdPane		_menu;
    CmdPane		_cmd;
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const XilImage<u_char>& image,
			 const IntArray2& points,
			 u_int resolution,
			 u_int underlayCmapDim, u_int overlayCmapDim)
    :CmdWindow(parentApp, name, 0, Colormap::IndexedColor,
		  resolution, underlayCmapDim, overlayCmapDim),
     _menu(*this, MainMenu), _cmd(*this, Cmds),
     _canvas(*this, image, points)
{
    _menu.place(0, 0, 1, 1);
    _cmd.place(0, 1, 1, 1);
    _canvas.place(0, 2, 1, 1);
    
    for (int i = 0; i < points.dim(); ++i)
	colormap().setOverlayValue(1+i, bgr[i]);

    show();
}

void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case M_Exit:
	app().exit();
	break;

      case c_Slider:
	_canvas.dc().setZoom(val, range[2]).repaintAll();
        break;
      
      case c_Underlay:
	_canvas._underlay = val;
	if (val)
	    _canvas.dc() << repaint;
	else
	    _canvas.dc() << clear;
	break;

      case c_Overlay:
	_canvas._overlay = val;
	if (val)
	    _canvas.dc() << overlay << repaint << underlay;
	else
	    _canvas.dc() << overlay << clear << underlay;
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
    
    v::App			vapp(argc, argv);
    TU::XilImage<u_char>	image;
    ifstream in("/home/ueshiba/data/pbm/ALV.pgm", ios::in);
    image.restore(in);

    IntArray2	points;
    cerr << "Points >> " << flush;
    cin >> points;

    v::MyCmdWindow	myWin0(vapp, "Image viewer (0)", image, points,
			       16, 8, 3);

    vapp.run();

    cerr << "Loop exited!" << endl;

    return 0;
}
