/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$  
 */
#include "TU/v/Menu.h"
#include "TU/v/Colormap.h"
#include "TU/v/Bitmap.h"
#include <X11/Xaw3d/SimpleMenu.h>
#include <X11/Xaw3d/SmeBSB.h>
#include <X11/Xaw3d/SmeLine.h>
#include <X11/Xaw3d/Sme.h>

namespace TU
{
namespace v
{
/************************************************************************
*  class Menu								*
************************************************************************/
Menu::Menu(Object& parentObject, const MenuDef menu[])
   // parent: MenuButtonCmd, ChoiceMenuButtonCmd or CanvasPaneDC
    :Cmd(parentObject, M_Default),
   // Widget name must be equal to the XtNmenuName resource of a
   // MenuPane::Button object.
     _widget(XtVaCreatePopupShell("TUvMenu",
				  simpleMenuWidgetClass,
				  parent().widget(),
				// This resource setting cannot be ommited.
				  XtNvisual,
				      window().colormap().vinfo().visual,
				  Null))
{
    for (int i = 0; menu[i].label != 0; ++i)
	addCmd(new Item(*this, menu[i]));
}

Menu::Menu(Object& parentObject, const MenuDef menu[],
		 const char* name, ::Widget parentWidget)
   // parent: Menu::Item
    :Cmd(parentObject, M_Default),
   // Widget name is set to the label of the parent menu item.
     _widget(XtVaCreatePopupShell(name,
				  simpleMenuWidgetClass,
				  parentWidget,
				// This resource setting cannot be ommited.
				  XtNvisual,
				      window().colormap().vinfo().visual,
				  Null))
{
    for (int i = 0; menu[i].label != 0; ++i)
	addCmd(new Item(*this, menu[i]));
}

Menu::~Menu()
{
    for (Cmd* vcmd; (vcmd = detachCmd()) != 0; )
	delete vcmd;
}

const Object::Widget&
Menu::widget() const
{
    return _widget;
}

/************************************************************************
*  bitmaps for Menu:Item						*
************************************************************************/
static u_char	checkBits[] =
		{
		    9, 9,
		    0xff, 0x01, 0x83, 0x01, 0x45, 0x01,
		    0x29, 0x01, 0x11, 0x01, 0x29, 0x01,
		    0x45, 0x01, 0x83, 0x01, 0xff, 0x01
		};
static Bitmap*	checkBitmap = 0;

static u_char	clearBits[] = {9,9,
			       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static Bitmap*	clearBitmap = 0;

static u_char	arrowBits[] =
		{
		    9, 9,
		    0x03, 0x00, 0x0f, 0x00, 0x3f, 0x00,
		    0xff, 0x00, 0xff, 0x01, 0xff, 0x00,
		    0x3f, 0x00, 0x0f, 0x00, 0x03, 0x00
		};
static Bitmap*	arrowBitmap = 0;

/************************************************************************
*  class Menu:Item							*
************************************************************************/
u_int			Menu::Item::_nitems = 0;

Menu::Item::Item(Menu& parentMenu, const MenuDef& menuItem)
    :Cmd(parentMenu, menuItem.id),
     _widget(menuItem.id == M_Line ?
	     XtVaCreateManagedWidget("-",
				     smeLineObjectClass,
				     parent().widget(),
				     XtNvertSpace,	0,
				     XtNleftMargin,	15,
				     XtNrightMargin,	15,
				     XtNbackground,
				     parent().widget().background(), 
				     Null) :
	     XtVaCreateManagedWidget("TUvMenu-Item",	//menuItem.label,
				     smeBSBObjectClass,
				     parent().widget(),
				     XtNlabel,		menuItem.label,
				     XtNvertSpace,	25,
				     XtNleftMargin,	15,
				     XtNrightMargin,	15,
				     XtNbackground,
				     parent().widget().background(), 
				     Null))
{
    if (_nitems++ == 0)
    {
	checkBitmap = new Bitmap(window().colormap(), checkBits);
	clearBitmap = new Bitmap(window().colormap(), clearBits);
	arrowBitmap = new Bitmap(window().colormap(), arrowBits);
    }

    if (menuItem.id != M_Line)
	if (menuItem.submenu != 0)
	{
	    if (menuItem.checked)  // If checked, submenu is a choice menu.
		addCmd(new ChoiceMenu(*this, menuItem.submenu,
				      menuItem.label,
				      parent().widget()));
	    else		   // Ordinary menu.
		addCmd(new Menu(*this, menuItem.submenu,
				menuItem.label, parent().widget()));
	    XtVaSetValues(_widget, XtNrightBitmap, arrowBitmap->xpixmap(),
			  Null);
	}
	else
	{
	    setValue(menuItem.checked);		// Show check mark.
	    setDefaultCallback(_widget);
	}
}

Menu::Item::~Item()
{
    delete detachCmd();

    if (--_nitems == 0)
    {
	delete arrowBitmap;
	delete clearBitmap;
	delete checkBitmap;
    }
}

const Object::Widget&
Menu::Item::widget() const
{
    return _widget;
}

void
Menu::Item::callback(CmdId nid, CmdVal nval)
{
    if (findChild(nid) != 0)
    {
	setValue(nval);
	parent().callback(id(), getValue());
    }
    else
	parent().callback(nid, nval);
}

CmdVal
Menu::Item::getValue() const
{
    const Cmd*	submenu = findChild(M_Default);
    if (submenu != 0)
	return submenu->getValue();
    else
    {
	Pixmap	bitmap;
	XtVaGetValues(_widget, XtNleftBitmap, &bitmap, Null);
	return (bitmap == checkBitmap->xpixmap() ? 1 : 0);
    }
}

void
Menu::Item::setValue(CmdVal val)
{
    Cmd*	submenu = findChild(M_Default);
    if (submenu != 0)
	submenu->setValue(val);
    else
	XtVaSetValues(_widget,
		      XtNleftBitmap, (val != 0 ? checkBitmap->xpixmap() :
						 clearBitmap->xpixmap()),
		      Null);
}

/************************************************************************
*  class ChoiceMenu							*
************************************************************************/
ChoiceMenu::ChoiceMenu(Object& parentObject, const MenuDef menu[])
    :Menu(parentObject, menu)
{
}

ChoiceMenu::ChoiceMenu(Object& parentObject, const MenuDef menu[],
		       const char* name, ::Widget parentWidget)
    :Menu(parentObject, menu, name, parentWidget)
{
}

ChoiceMenu::~ChoiceMenu()
{
}

void
ChoiceMenu::callback(CmdId nid, CmdVal nval)
{
    if (findChild(nid) != 0)
    {
	setValue(nid);
	parent().callback(id(), getValue());
    }
    else
	parent().callback(nid, nval);
}

CmdVal
ChoiceMenu::getValue() const
{
    const Cmd*	vcmd = findChildWithNonZeroValue();
    return (vcmd != 0 ? vcmd->id() : M_None);
}

void
ChoiceMenu::setValue(CmdVal val)
{
    Cmd*	citem = findChild(getValue());	// current checked item
    Cmd*	nitem = findChild(val);		// next checked item
    if (nitem != citem)
    {
	if (citem != 0)
	    citem->setValue(false);
	if (nitem != 0)
	    nitem->setValue(true);
    }
}

}
}
