/*
 *  $Id: main.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include <fstream>
#include "TU/v/App.h"
#include "TU/v/CmdPane.h"
#include "TU/v/XglDC.h"

namespace TU
{
namespace v
{
/************************************************************************
*  menus and commands							*
************************************************************************/
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

/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
class MyCanvasPane : public CanvasPane
{
  public:
    MyCanvasPane(Window& parentWin, const Image<BGR>& image)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _dc(*this), _image(image)					{}

    virtual void	repaintUnderlay(int x, int y, int w, int h)	;

  protected:
    virtual void	initializeGraphics()				;
    
  private:
    XglDC		_dc;
    const Image<BGR>&	_image;
};

void
MyCanvasPane::repaintUnderlay(int, int, int, int)
{
    static const float	CX = -32.0, CY = 128.0, CZ = -16.0,
			LX =  64.0, LY =  16.0, LZ =  32.0;

    _dc << clearXgl << _image;
    
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
  //    xglcolor.index = 1;
    xglcolor.rgb.r = 1.0;
    xglcolor.rgb.g = 0.0;
    xglcolor.rgb.b = 0.0;
    xgl_object_set(_dc, XGL_CTX_LINE_COLOR, &xglcolor, NULL);
    xgl_multipolyline(_dc, NULL, sizeof(pl)/sizeof(pl[0]), pl);
    _dc << syncXgl;
}

void
MyCanvasPane::initializeGraphics()
{
}

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&	parentApp, const char* name,
		const Image<BGR>& image)			;

    virtual void	callback(CmdId, CmdVal)			;
    
  private:
    CmdPane		_menu;
    MyCanvasPane	_canvas;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const Image<BGR>& image)
    :CmdWindow(parentApp, name, 0, Colormap::RGBColor, 16, 0, 0),
     _menu(*this, MainMenu), _canvas(*this, image)
{
    _menu.place(0, 0, 1, 1);
    _canvas.place(0, 1, 1, 1);
    
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
    Image<BGR>		image;
    ifstream		in("/home/ueshiba/data/pbm/harumi.ppm", ios::in);
    image.restore(in);
    
    try
    {
	v::MyCmdWindow	myWin0(vapp, "XGL test", image);
	vapp.run();
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }

    cerr << "Loop exited!" << endl;

    return 0;
}
