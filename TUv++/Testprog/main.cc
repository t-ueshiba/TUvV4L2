/*
 *  $Id: main.cc,v 1.3 2002-12-18 06:10:13 ueshiba Exp $
 */
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include "TU/v/App.h"
#include "TU/v/ModalDialog.h"
#include "TU/v/CmdWindow.h"
#include "TU/v/CmdPane.h"
#include "TU/v/CanvasPane.h"
#include "TU/v/CanvasPaneDC.h"
#include "TU/v/ShmDC.h"

namespace TU
{
typedef Array2<Array<int> >	IntArray2;

namespace v
{
/************************************************************************
*  menus and commands							*
************************************************************************/
enum	{
	    m_X05, m_X10, m_X15, m_X20, m_Zoom,
	    c_Slider,
	    c_Button, c_Underlay, c_Overlay, c_ChoiceFrame,
	    c_Radio, c_Radio1, c_Radio2, c_List, c_TextIn, c_Dump
	};

static MenuDef ZoomMenu[] =
{
    {"x0.5", m_X05, false, noSub},
    {"x1",   m_X10, true,  noSub},
    {"x1.5", m_X15, false, noSub},
    {"x2",   m_X20, false, noSub},
    EndOfMenu
};

static MenuDef FileMenu[] =
{
    {"New",  M_New,  false, noSub},
    {"Open", M_Open, false, noSub},
    {"Zoom", m_Zoom, true,  &ZoomMenu[0]},
    {"-",    M_Line, false, noSub},
    {"Quit", M_Exit, false, noSub},
    EndOfMenu
};

static CmdDef MainMenu[] =
{
    {C_MenuButton,	 M_File, 0, "File", FileMenu, CA_None, 0, 0, 1, 1, 0},
    {C_ChoiceMenuButton, m_Zoom, m_X10, "Zoom", ZoomMenu, CA_None, 1, 0, 1, 1, 0},
    EndOfCmds
};

static int  range[] = {0, 100, 50};

static CmdDef subCmds[] =
{
  {C_RadioButton, c_Radio,  0, "Radio",  noProp, CA_None, 0, 0, 1, 1, 0},
  {C_RadioButton, c_Radio1, 0, "Radio1", noProp, CA_None, 1, 0, 1, 1, 0},
  {C_RadioButton, c_Radio2, 0, "Radio2", noProp, CA_None, 2, 0, 1, 1, 0},
  EndOfCmds
};

static const char*	listItem[] =
{
    "Item 0", "Item 1", "Item 2", "Item 3", "Item 4", "Item 5", "Item 6", NULL
};

static CmdDef Cmds[] =
{
  {C_Slider,       c_Slider,  50, "Zoom:",    range,   CA_None, 0, 0, 1, 1, 0},
  {C_Button,	   c_Button,   0, "Dialog",   noProp,  CA_None, 1, 0, 1, 1, 0},
  {C_ToggleButton, c_Underlay, 1, "Underlay", noProp,  CA_None, 2, 0, 1, 1, 0},
  {C_ToggleButton, c_Overlay,  1, "Overlay",  noProp,  CA_None, 3, 0, 1, 1, 0},
  {C_Button,	   c_Dump,     0, "Dump",     noProp,  CA_None, 4, 0, 1, 1, 0},
  {C_ChoiceFrame,  c_ChoiceFrame, c_Radio, "",subCmds, CA_None, 0, 1, 1, 1, 0},
  {C_List,	   c_List,     0, "",	      listItem,CA_None, 1, 1, 2, 1, 3},
  {C_TextIn,	   c_TextIn,   0, "TextIn",   noProp,  CA_None, 3, 1, 2, 1,10},
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
    MyCanvasPane(Window& parentWin, const Image<u_char>& image,
		 const IntArray2& points)
	:CanvasPane(parentWin, image.width(), image.height()),
	 _dc(*this),
	 _image(image), _points(points), _underlay(1), _overlay(1)	{}

    CanvasPaneDC&	dc()					{return _dc;}
    
    virtual void	repaintUnderlay(int x, int y, int w, int h)	;
    virtual void	repaintOverlay(int x, int y, int w, int h)	;

  private:
    CanvasPaneDC		_dc;
  //  ShmDC			_dc;
    const Image<u_char>&	_image;
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
*  class MyDialog							*
************************************************************************/
class MyDialog : public ModalDialog
{
  public:
    MyDialog(Window& parentWindow, const char* myName,
	     const CmdDef cmd[])					;

    virtual void	callback(CmdId id, CmdVal val)		;
};

MyDialog::MyDialog(Window& parentWindow, const char* myName,
		   const CmdDef cmd[])
    :ModalDialog(parentWindow, myName, cmd)
{
}

void
MyDialog::callback(CmdId id, CmdVal val)
{
    using namespace	std;
	
    switch (id)
    {
      case c_Slider:
	cerr << "MyDialog::windowCommand: slider moved   (val: " << val << ")"
	     << endl;
        break;
      
      case c_Button:
	cerr << "MyDialog::windowCommand: button pressed (val: " << val << ")"
	     << endl;
	hide();
	break;
    }
}

/************************************************************************
*  class MyCmdWindow							*
************************************************************************/
class MyCmdWindow : public CmdWindow
{
  public:
    MyCmdWindow(App&			parentApp,
		const char*		name,
		const Image<u_char>&	image,
		const IntArray2&	points)			;

    virtual void	callback(CmdId, CmdVal)			;
    
  private:
    CmdPane		_menu;
    CmdPane		_cmd;
    MyCanvasPane	_canvas0, _canvas1;
    MyDialog		_dialog;
};

MyCmdWindow::MyCmdWindow(App& parentApp, const char* name,
			 const Image<u_char>& image,
			 const IntArray2& points)
    :CmdWindow(parentApp, name, 0, Colormap::IndexedColor, 16, 8, 3),
     _menu(*this, MainMenu), _cmd(*this, Cmds),
     _canvas0(*this, image, points),
     _canvas1(*this, image, points),
     _dialog(*this, "My Dialog", Cmds)
{
    _menu.place(0, 0, 2, 1);
    _cmd.place(0, 1, 2, 1);
    _canvas0.place(0, 2, 1, 1);
    _canvas1.place(1, 2, 1, 1);
    
    for (int i = 0; i < points.nrow(); ++i)
	colormap().setOverlayValue(1+i, bgr[i]);

    show();
}

void
MyCmdWindow::callback(CmdId id, CmdVal val)
{
    using namespace	std;
	
    switch (id)
    {
      case Id_MouseButton1Press:
	cerr << "Button1 pressed at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton2Press:
	cerr << "Button2 pressed at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton3Press:
	cerr << "Button3 pressed at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton1Release:
	cerr << "Button1 released at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton2Release:
	cerr << "Button2 released at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton3Release:
	cerr << "Button3 released at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton1Drag:
	cerr << "Button1 dragged at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton2Drag:
	cerr << "Button2 dragged at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseButton3Drag:
	cerr << "Button3 dragged at ("
	     << val.u << ", " << val.v << ")." << endl;
	break;
	
      case Id_MouseMove:
	cerr << "Mouse moved to (" << val.u << ", " << val.v << ")." << endl;
	break;

      case Id_MouseEnterFocus:
	cerr << "Enter focus." << endl;
	break;
	
      case Id_MouseLeaveFocus:
	cerr << "Leave focus." << endl;
	break;

      case Id_KeyPress:
	if (val & VKM_Ctrl)
	    cerr << "Ctrl-";
	if (val & VKM_Alt)
	    cerr << "Alt-";
	cerr << char(val & ~VKM_Ctrl & ~VKM_Alt) << " pressed." << endl;
	break;

      case M_Exit:
	app().exit();
	break;

      case m_Zoom:
	switch (val)
	{
	  case m_X05:
	    _canvas0.dc() << x0_5;
	    _menu.setValue(m_Zoom, m_X05);
	    _cmd.setValue(c_Slider, 0.5f);
	    break;
	  case m_X10:
	    _canvas0.dc() << x1;
	    _menu.setValue(m_Zoom, m_X10);
	    _cmd.setValue(c_Slider, 1.0f);
	    break;
	  case m_X15:
	    _canvas0.dc() << x1_5;
	    _menu.setValue(m_Zoom, m_X15);
	    _cmd.setValue(c_Slider, 1.5f);
	    break;
	  case m_X20:
	    _canvas0.dc() << x2;
	    _menu.setValue(m_Zoom, m_X20);
	    _cmd.setValue(c_Slider, 2.0f);
	    break;
	}
	_canvas0.dc() << repaintAll;
	break;

      case c_Slider:
	_canvas0.dc().setZoom(val, range[2]).repaintAll();
	break;
      
      case c_Button:
	cerr << "Button pressed (val: " << val << ")" << endl;
	_dialog.show();
	cerr << "Dialog done." << endl;
	break;

      case c_Underlay:
	_canvas0._underlay = val;
	if (val)
	    _canvas0.dc() << repaint;
	else
	    _canvas0.dc() << clear;
	break;

      case c_Overlay:
	_canvas0._overlay = val;
	if (val)
	    _canvas0.dc() << overlay << repaint << underlay;
	else
	    _canvas0.dc() << overlay << clear << underlay;
	break;

      case c_Dump:
      {
	ofstream out("tmp.xwd", ios::out);
	_canvas0.dc().dump(out);
      }
	break;

      case c_ChoiceFrame:
	cerr << "Choice frame (button: " << val << ")" << endl;
	break;
	
      case c_List:
	cerr << "List pressed (item: " << val << ")" << endl;
	break;

      case c_TextIn:
	cerr << "Text input (str: " << _cmd.getString(c_TextIn) << ")"
	     << endl;
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
    using namespace	TU;
    using namespace	std;

    v::App		vapp(argc, argv);
    Image<u_char>	image;
    ifstream		in("/home/ueshiba/data/pbm/ALV.pgm", ios::in);
    image.restore(in);

    try
    {
	IntArray2	points;
	cerr << "Points >> " << flush;
	cin >> points;

	v::MyCmdWindow	myWin0(vapp, "Image viewer (0)", image, points);
  //    v::MyCmdWindow	myWin1(vapp, "Image viewer (1)", image, points);

	vapp.run();
    }
    catch (exception& err)
    {
        cerr << err.what() << endl;
	return 1;
    }

    cerr << "Loop exited!" << endl;

    return 0;
}

#ifdef __GNUG__
#  include "TU/Array++.cc"
#endif
