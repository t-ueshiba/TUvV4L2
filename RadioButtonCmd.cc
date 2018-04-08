/*
 *  $Id$  
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
size_t		RadioButtonCmd::_nitems = 0;

RadioButtonCmd::RadioButtonCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvRadioButtonCmd", cmd),
     _button(XtVaCreateManagedWidget("TUvRadioButtonCmd-button",
				     commandWidgetClass,
				     _widget,
				     XtNbackground,
				     parent().widget().background(),
				     XtNrelief,			XtReliefNone,
				     XtNborderWidth,		0,
				     nullptr)),
     _label(XtVaCreateManagedWidget("TUvRadioButtonCmd-label",
				    labelWidgetClass,
				    _widget,
				    XtNbackground,
				    parent().widget().background(),
				    XtNlabel,			cmd.title,
				    XtNinternalHeight,		3,
				    XtNinternalWidth,		0,
				    XtNborderWidth,		0,
				    XtNrelief,			XtReliefNone,
				    XtNhighlightThickness,	0,
				    XtNgridx,			1,
				    nullptr))
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
    removeDefaultCallback(_button);

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
    XtVaGetValues(_button, XtNbitmap, &bitmap, nullptr);
    return (bitmap == onBitmap->xpixmap() ? 1 : 0);
}

void
RadioButtonCmd::setValue(CmdVal val)
{
    XtVaSetValues(_button,
		  XtNbitmap,	(val != 0 ?
				 onBitmap->xpixmap() : offBitmap->xpixmap()),
		  nullptr);
}

}
}
