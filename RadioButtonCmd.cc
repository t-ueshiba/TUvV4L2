/*
 *  $BJ?@.(B9-19$BG/!JFH!K;:6H5;=QAm9g8&5f=j(B $BCx:n8"=jM-(B
 *  
 *  $BAO:n<T!'?"<G=SIW(B
 *
 *  $BK\%W%m%0%i%`$O!JFH!K;:6H5;=QAm9g8&5f=j$N?&0w$G$"$k?"<G=SIW$,AO:n$7!$(B
 *  $B!JFH!K;:6H5;=QAm9g8&5f=j$,Cx:n8"$r=jM-$9$kHkL)>pJs$G$9!%AO:n<T$K$h(B
 *  $B$k5v2D$J$7$KK\%W%m%0%i%`$r;HMQ!$J#@=!$2~JQ!$;HMQ!$Bh;0<T$X3+<($9$k(B
 *  $BEy$NCx:n8"$r?/32$9$k9T0Y$r6X;_$7$^$9!%(B
 *  
 *  $B$3$N%W%m%0%i%`$K$h$C$F@8$8$k$$$+$J$kB;32$KBP$7$F$b!$Cx:n8"=jM-<T$*(B
 *  $B$h$SAO:n<T$O@UG$$rIi$$$^$;$s!#(B
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: RadioButtonCmd.cc,v 1.3 2007-11-26 08:11:50 ueshiba Exp $
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
