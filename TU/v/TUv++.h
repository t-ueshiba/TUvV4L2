/*
 *  $Id$  
 */
#ifndef TU_VPP_H
#define TU_VPP_H

#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include "TU/List.h"
#include "TU/Geometry++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  Global type definitions						*
************************************************************************/
typedef int	CmdId;			// ID for each item command or menu

class CmdVal
{
  public:
    CmdVal(int u=0, int v=0)	:_u(u), _v(v), _f(u)		{}
    template <class T,
	      class=typename std::enable_if<std::is_arithmetic<T>::value>::type>
    CmdVal(T val)		:_u(int(val)), _v(0), _f(val)	{}
    
		operator int()				const	{return _u;}
    int		u()					const	{return _u;}
    int		v()					const	{return _v;}
    float	f()					const	{return _f;}
	
  private:
    int		_u;
    int		_v;
    float	_f;
};

/************************************************************************
*  struct CmdType							*
************************************************************************/
enum CmdType	// types of dialog commands
{
    C_EndOfList,		// Used to denote end of command list
    C_Button,			// Button
    C_ChoiceFrame,		// Choice frame
    C_ChoiceMenuButton,		// A choice menu button
    C_Frame,			// General purpose frame
    C_Icon,			// A display only Icon
    C_Label,			// Regular text label
    C_List,			// List of items (scrollable)
    C_Menu,			// Menu
    C_MenuItem,			// Menu item
    C_MenuButton,		// Menu button
    C_RadioButton,		// Radio button
    C_Slider,			// Slider to enter value
    C_TextIn,			// Text input field
    C_ToggleButton,		// A toggle button

    C_ZZZ			// make my life easy
};

enum CmdAttributes
{
    CA_None	     =      0,	// No special attributes
    CA_DefaultButton =   0x02,	// Special Default Button
    CA_Horizontal    =   0x20,	// Horizontal orentation
    CA_Vertical	     =   0x40,	// Vertical orentation
    CA_NoBorder	     =  0x400,	// No border(frames,status bar)
    CA_NoSpace	     =  0x800	// No space between widgets
};

struct CmdDef
{
    CmdDef(CmdType	type_=C_EndOfList,
	   CmdId	id_=0,
	   CmdVal	val_=0,
	   const char*	title_=nullptr,
	   void*	prop_=nullptr,
	   u_int	attrs_=CA_None,
	   size_t	gridx_=0,
	   size_t	gridy_=0,
	   size_t	gridWidth_=1,
	   size_t	gridHeight_=1,
	   size_t	size_=0)
	:type(type_), id(id_), val(val_), title(title_),
	 prop(prop_), attrs(attrs_), gridx(gridx_), gridy(gridy_),
	 gridWidth(gridWidth_), gridHeight(gridHeight_), size(size_)	{}
	 
    CmdType	type;		// what kind of item is this
    CmdId	id;		// unique id for the item
    CmdVal	val;		// value returned when picked (might be < 0)
    const char*	title;		// string
    void*	prop;		// a list of stuff to use for the cmd
    u_int	attrs;		// bit map of attributes
    size_t	gridx;		// x-position in the parent grid.
    size_t	gridy;		// y-position in the parent grid.
    size_t	gridWidth;	// width  in # of cells of the parent grid.
    size_t	gridHeight;	// height in # of cells of the parent grid.
    size_t	size;		// width in pixels
};

void* const	noProp	= nullptr;

#define EndOfCmds	{C_EndOfList, 0, 0, 0, noProp, CA_None, 0, 0, 1, 1, 0}

/************************************************************************
*  struct MenuDef							*
************************************************************************/
struct MenuDef
{
    MenuDef(const char*	label_=nullptr,
	    CmdId	id_=0,
	    bool	checked_=false,
	    MenuDef*	submenu_=nullptr)
	:label(label_), id(id_), checked(checked_), submenu(submenu_)	{}
    
    const char*	label;
    CmdId	id;
    bool	checked;
    MenuDef*	submenu;
};

MenuDef* const	noSub			= nullptr;

// standard menu item definitions
const CmdId	M_File			= 32000;
const CmdId	M_Edit			= 32001;
const CmdId	M_Search		= 32002;
const CmdId	M_Help			= 32003;
const CmdId	M_Window		= 32004;
const CmdId	M_Options		= 32005;
const CmdId	M_Tools			= 32006;
const CmdId	M_Font			= 32007;
const CmdId	M_View			= 32008;
const CmdId	M_Format		= 32009;
const CmdId	M_Insert		= 32010;
const CmdId	M_Test			= 32011;

const CmdId	M_Line			= 32090;

const CmdId	M_New			= 32100;
const CmdId	M_Open			= 32101;
const CmdId	M_Close			= 32102;
const CmdId	M_Save			= 32103;
const CmdId	M_SaveAs		= 32104;
const CmdId	M_Print			= 32105;
const CmdId	M_PrintPreview		= 32106;
const CmdId	M_About			= 32107;
const CmdId	M_Exit			= 32108;
const CmdId	M_CloseFile		= 32109;	// V:1.13

const CmdId	M_Cut			= 32110;
const CmdId	M_Copy			= 32111;
const CmdId	M_Paste			= 32112;
const CmdId	M_Delete		= 32113;
const CmdId	M_Clear			= 32114;

const CmdId	M_UnDo			= 32120;
const CmdId	M_SetDebug		= 32121;

const CmdId	M_Find			= 32130;
const CmdId	M_FindAgain		= 32131;
const CmdId	M_Replace		= 32132;

const CmdId	M_Preferences		= 32140;
const CmdId	M_FontSelect		= 32141;

const CmdId	M_Cancel		= 32150;
const CmdId	M_Done			= 32151;
const CmdId	M_OK			= 32152;
const CmdId	M_Yes			= 32153;
const CmdId	M_No			= 32154;
const CmdId	M_All			= 32155;
const CmdId	M_None			= 32156;
const CmdId	M_Default		= 32157;

#define EndOfMenu	{0, 0, false, noSub}

/************************************************************************
*  IDs for Mouse/Key							*
************************************************************************/
const CmdId	Id_MouseButton1Press	= 32200;  // mouse button pressed
const CmdId	Id_MouseButton2Press	= 32201;
const CmdId	Id_MouseButton3Press	= 32202;
const CmdId	Id_MouseButton1Release	= 32203;  // mouse button released
const CmdId	Id_MouseButton2Release	= 32204;
const CmdId	Id_MouseButton3Release	= 32205;
const CmdId	Id_MouseButton1Drag	= 32206;  // mouse dragged
const CmdId	Id_MouseButton2Drag	= 32207;
const CmdId	Id_MouseButton3Drag	= 32208;
const CmdId	Id_MouseMove		= 32209;  // mouse moved
const CmdId	Id_MouseEnterFocus	= 32210;  // mouse entered the window
const CmdId	Id_MouseLeaveFocus	= 32211;  // mouse leaved the window
const CmdId	Id_KeyPress		= 32300;  // mouse leaved the window

// key modifiers
const int	VKM_Ctrl		= 0x100;
const int	VKM_Alt			= 0x200;

/************************************************************************
*  class Object								*
************************************************************************/
class App;
class Window;
class CanvasPane;
class Timer;
class Object
{
  public:
#ifdef UseGtk
#  include "TU/v/Widget-Gtk+.h"
#else
#  include "TU/v/Widget-Xaw.h"
#endif

  public:
    virtual const Widget&	widget()		const	= 0;

    virtual void		callback(CmdId id, CmdVal val)	;
    virtual void		tick()				;
    
  protected:
    Object(Object& parentObject) :_parent(parentObject)	{}
    virtual ~Object()					;

	    Object&	parent()			{return _parent;}

    virtual App&	app()				;
    virtual Window&	window()			;
    virtual CanvasPane&	canvasPane()			;

    
  private:
    Object(const Object&)				;
    Object&		operator =(const Object&)	;

    Object&		_parent;

    friend class	Window;			// for accessing window()
    friend class	Timer;			// for accessing app()
};

/************************************************************************
*  class Pane								*
************************************************************************/
class Pane : public Object, public List<Pane>::Node
{
  public:
    Pane(Window& parentWin)					;
    virtual	 ~Pane()					;

    void	place(size_t gridx=0, size_t gridy=0,
		      size_t gridWidth=1, size_t gridHeight=1)	;
};

/************************************************************************
*  class Window								*
************************************************************************/
class Colormap;
class Window : public Object, public List<Window>::Node
{
  public:
    virtual Colormap&	colormap()				;
	
    virtual void	show()					;
    virtual void	hide()					;
	    bool	isFullScreen()			const	;

  protected:
    Window(Window& parentWindow)				;
    virtual		 ~Window()				;
    
    virtual Window&	window()				;
    virtual CanvasPane&	canvasPane()				;
    
	    void	addWindow(Window& win)			;
	    void	detachWindow(Window& win)		;

    virtual Object&	paned()					;

  private:
    friend class	Pane;		 /* allow access to addPane(),
					    detachPane() and paned(). */
	    void	addPane(Pane& pane)			;
	    void	detachPane(Pane& pane)			;
    
    List<Window>	_windowList;
    List<Pane>		_paneList;
};

inline void
Window::addWindow(Window& win)
{
    _windowList.push_front(win);
}

inline void
Window::detachWindow(Window& win)
{
    _windowList.remove(win);
}

inline void
Window::addPane(Pane& pane)
{
    _paneList.push_front(pane);
}

inline void
Window::detachPane(Pane& pane)
{
    _paneList.remove(pane);
}

/************************************************************************
*  class CmdParent							*
************************************************************************/
class Cmd;
class CmdParent
{
  public:
    CmdVal	getValue(CmdId id)			const	;
    void	setValue(CmdId id, CmdVal val)			;
    const char*	getString(CmdId id)			const	;
    void	setString(CmdId id, const char* str)		;
    void	setProp(CmdId id, const void* prop)		;
    
  protected:
    void	addCmd(Cmd* vcmd)				;
    Cmd*	detachCmd()					;
    const Cmd*	findChild(CmdId id)			const	;
	  Cmd*	findChild(CmdId id)				;
    const Cmd*	findChildWithNonZeroValue()		const	;
    
  private:
    const Cmd*	findDescendant(CmdId id)		const	;
	  Cmd*	findDescendant(CmdId id)			;

    List<Cmd>	_cmdList;
};

/************************************************************************
*  class Cmd								*
************************************************************************/
class Cmd : public Object, public CmdParent, public List<Cmd>::Node
{
  public:
    static Cmd*		newCmd(Object& parentObject, const CmdDef& cmd)	;
    virtual		~Cmd()						;
    
    CmdId		id()				const	{return _id;}
    virtual CmdVal	getValue()			const	;
    virtual void	setValue(CmdVal val)			;
    virtual const char*	getString()			const	;
    virtual void	setString(const char* str)		;
    virtual void	setProp(const void* prop)		;
     
  protected:
    Cmd(Object& parentObject, CmdId id)				;

    void	setDefaultCallback(const Widget& widget)	;
    void	removeDefaultCallback(const Widget& widget)	;
     
  private:
    const CmdId		_id;			// unique ID for this command
};

inline void
CmdParent::addCmd(Cmd* vcmd)
{
    _cmdList.push_front(*vcmd);
}

inline Cmd*
CmdParent::detachCmd()
{
    return (_cmdList.empty() ? nullptr : &_cmdList.pop_front());
}

}	// namespace v
}	// namespace TU
#endif	// ! TU_VPP_H
