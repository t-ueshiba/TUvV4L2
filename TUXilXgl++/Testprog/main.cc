/*
 *  $Id: main.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include <iomanip>
#include <fstream>
#include "TU/v/App.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/XilXglDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  menus and commands							*
************************************************************************/
enum	{c_Underlay=100, c_Overlay};

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

static CmdDef Cmds[] =
{
  {C_ToggleButton, c_Underlay, 1, "Underlay", noProp, CA_None, 0, 0, 1, 1, 0},
  {C_ToggleButton, c_Overlay,  1, "Overlay",  noProp, CA_None, 1, 0, 1, 1, 0},
  EndOfCmds
};

/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const XilImage<u_char>& image)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _dc(*this), _image(image), _underlay(1), _overlay(1)		{}

    XilXglDC&	dc()				{return _dc;}
    
    virtual void	repaintUnderlay(int x, int y, int w, int h)	;
    virtual void	repaintOverlay(int x, int y, int w, int h)	;

  protected:
    virtual void	initializeGraphics()				;
    
  private:
    XilXglDC			_dc;
    const XilImage<u_char>&	_image;

  public:
    int				_underlay, _overlay;
};

void
MyCanvasPane::repaintUnderlay(int, int, int, int)
{
    if (!_underlay)
	return;
    
    _dc << clear << _image << sync;
}

void
MyCanvasPane::repaintOverlay(int, int, int, int)
{
    if (!_overlay)
	return;
    
    static const float	CX = -32.0, CY = 128.0, CZ = -16.0,
			LX =  64.0, LY =  16.0, LZ =  32.0;

    Xgl_pt_f3d		pt[4];
    pt[0].x = CX;	pt[0].y = CY;		pt[0].z = CZ;
    pt[1].x = CX + LX;	pt[1].y = CY;		pt[1].z = CZ;
    pt[2].x = CX;	pt[2].y = CY + LY;	pt[2].z = CZ;
    pt[3].x = CX;	pt[3].y = CY;		pt[3].z = CZ + LZ;

    Xgl_pt_list		pl[1];
    pl[0].pt_type	  = XGL_PT_F3D;
    pl[0].bbox		  = NULL;
    pl[0].num_pts	  = sizeof(pt)/sizeof(pt[0]);
    pl[0].num_data_values = 0;
    pl[0].pts.f3d	  = pt;

    Xgl_color	xglcolor;
    xglcolor.index = 1;
  /*    xglcolor.rgb.r = 1.0;
    xglcolor.rgb.g = 1.0;
    xglcolor.rgb.b = 0.0;*/
    xgl_object_set(_dc, XGL_CTX_LINE_COLOR, &xglcolor, NULL);
    _dc << clearXgl;
    xgl_multipolyline(_dc, NULL, sizeof(pl)/sizeof(pl[0]), pl);
    _dc << syncXgl;
}

void
MyCanvasPane::initializeGraphics()
{
    _dc.setInternal(_image.width() / 2.0, _image.height() / 2.0, 400.0, 400.0,
		    1.0, 1000.0);
    _dc << overlay;
}
 
/************************************************************************
*  class MyCmdWindow							*
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

class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App& parentApp, const char* name,
		const XilImage<u_char>& image)		;

    virtual void	callback(CmdId, CmdVal)			;
    
  private:
    CmdPane		_menu;
    CmdPane		_cmd;
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const XilImage<u_char>& image)
    :CmdWindow(parentApp, name, 0, Colormap::IndexedColor,
		  16, 8, 3),
     _menu(*this, MainMenu), _cmd(*this, Cmds), _canvas(*this, image)
{
    for (int i = 0; i < sizeof(bgr)/sizeof(bgr[0]); ++i)
	colormap().setUnderlayValue(1+i, bgr[i]);
    for (int i = 0; i < sizeof(bgr)/sizeof(bgr[0]); ++i)
	colormap().setOverlayValue(1+i, bgr[i]);

    _menu.place(0, 0, 1, 1);
    _cmd.place(0, 1, 1, 1);
    _canvas.place(0, 2, 1, 1);
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

      case c_Underlay:
	_canvas._underlay = val;
	if (val)
	    _canvas.dc() << underlay << repaint << overlay;
	else
	    _canvas.dc() << underlay << clear << overlay;
	break;

      case c_Overlay:
	_canvas._overlay = val;
	if (val)
	    _canvas.dc() << repaint;
	else
	    _canvas.dc() << clear;
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
    ifstream			in("/home/ueshiba/data/pbm/ALV.pgm", ios::in);
    image.restore(in);
    
    v::MyCmdWindow		myWin0(vapp, "XGL test", image);
    vapp.run();

    cerr << "Loop exited!" << endl;

    return 0;
}
