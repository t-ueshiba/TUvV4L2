/*
 *  $Id: RadioButtonCmd.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/Bitmap.h"
#include "RadioButtonCmd_.h"
#include <X11/Xaw3d/Command.h>
#include <X11/Xaw3d/ThreeD.h>
#include "vGridbox_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  bitmaps for RadioButtonCmd						*
************************************************************************/
static u_char	onBits[] =
		{
		    11, 11,
		    0x70, 0x00, 0x8c, 0x01, 0x72, 0x02,
		    0xfa, 0x02, 0xfd, 0x05, 0xfd, 0x05,
		    0xfd, 0x05, 0xfa, 0x02, 0x72, 0x02,
		    0x8c, 0x01, 0x70, 0x00
		};
static Bitmap*	onBitmap = 0;

static u_char	offBits[] =
		{
		    11, 11,
		    0x70, 0x00, 0x8c, 0x01, 0x02, 0x02,
		    0x02, 0x02, 0x01, 0x04, 0x01, 0x04,
		    0x01, 0x04, 0x02, 0x02, 0x02, 0x02,
		    0x8c, 0x01, 0x70, 0x00
		};
static Bitmap*	offBitmap = 0;

/************************************************************************
*  class RadioButtonCmd							*
************************************************************************/
u_int		RadioButtonCmd::_nitems = 0;

RadioButtonCmd::RadioButtonCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvRadioButtonCmd", cmd),
     _button(XtVaCreateManagedWidget("TUvRadioButtonCmd-button",
				     commandWidgetClass,
				     _widget,
				     XtNbackground,
				     parent().widget().background(),
				     XtNshadowWidth,		0,
				     XtNborderWidth,		0,
				     NULL)),
     _label(XtVaCreateManagedWidget("TUvRadioButtonCmd-label",
				    labelWidgetClass,
				    _widget,
				    XtNbackground,
				    parent().widget().background(),
				    XtNlabel,			cmd.title,
				    XtNinternalHeight,		3,
				    XtNinternalWidth,		0,
				    XtNborderWidth,		0,
				    XtNhighlightThickness,	0,
				    XtNgridx,			1,
				    NULL))
{
    if (_nitems++ == 0)
    {
	onBitmap  = new Bitmap(window().colormap(), onBits);
	offBitmap = new Bitmap(window().colormap(), offBits);
    }
	
    setValue(cmd.val);
    setDefaultCallback(_button);
}

RadioButtonCmd::~RadioButtonCmd()
{
    if (--_nitems == 0)
    {
	delete offBitmap;
	delete onBitmap;
    }
}

const Object::Widget&
RadioButtonCmd::widget() const
{
    return _widget;
}

CmdVal
RadioButtonCmd::getValue() const
{
    Pixmap	bitmap;
    XtVaGetValues(_button, XtNbitmap, &bitmap, NULL);
    return (bitmap == onBitmap->xpixmap() ? 1 : 0);
}

void
RadioButtonCmd::setValue(CmdVal val)
{
    XtVaSetValues(_button,
		  XtNbitmap,	(val != 0 ?
				 onBitmap->xpixmap() : offBitmap->xpixmap()),
		  NULL);
}

}
}
