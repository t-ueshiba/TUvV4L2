/*
 *  $Id: ListCmd.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "ListCmd_.h"
#include "vViewport_.h"
#include <X11/Xaw3d/List.h>
#include <X11/Xaw3d/Scrollbar.h>

namespace TU
{
namespace v
{
/************************************************************************
*  callbacks for scrollbarWidget					*
************************************************************************/
static void
CBjumpProc(::Widget, XtPointer This, XtPointer percent)
{
    ListCmd*	vListCmd = (ListCmd*)This;
    vListCmd->setPercent(*(float*)percent);
}

static void
CBscrollProc(::Widget widget, XtPointer This, XtPointer position)
{
    ListCmd*	vListCmd = (ListCmd*)This;
    if ((long)position > 0)
	vListCmd->scroll(1);
    else
	vListCmd->scroll(-1);
}

/************************************************************************
*  class ListCmd							*
************************************************************************/
ListCmd::ListCmd(Object& parentObject, const CmdDef& cmd)
    :Cmd(parentObject, cmd.id),
     _widget(parent().widget(), "TUvListCmd", cmd),
     _list(XtVaCreateManagedWidget("TUvListCmd-list",
				   listWidgetClass,
				   _widget,
				   XtNlist,		cmd.prop,
				   XtNdefaultColumns,	1,
				   XtNforceColumns,	TRUE,
				   XtNverticalList,	TRUE,
				   XtNinternalHeight,	0,
				   NULL)),
     _top(0),
     _nitems(0),
     _nitemsShown(cmd.size > 0 ? cmd.size : 10)
{
    setProp(cmd.prop);
    setValue(cmd.val);
    setDefaultCallback(_list);

    ::Widget	scrollbar = XtNameToWidget(_widget, "vertical");
    XtAddCallback(scrollbar, XtNjumpProc, CBjumpProc, this);
    XtAddCallback(scrollbar, XtNscrollProc, CBscrollProc, this);
}

ListCmd::~ListCmd()
{
}

const Object::Widget&
ListCmd::widget() const
{
    return _widget;
}

CmdVal
ListCmd::getValue() const
{
    XawListReturnStruct*	item = XawListShowCurrent(_list);
    return (item->list_index != XAW_LIST_NONE ? item->list_index : -1);
}

void
ListCmd::setValue(CmdVal val)
{
    XawListUnhighlight(_list);
    if (val >= 0)
	XawListHighlight(_list, val);
}

void
ListCmd::setProp(void* prop)
{
    if (prop != 0)
    {
	String*	s = (String*)prop;
	for (_nitems = 0; s[_nitems] != 0; )
	    ++_nitems;
	XawListChange(_list, (String*)prop, 0, 0, TRUE);

	Dimension	height;
	XtVaGetValues(_list, XtNheight, &height, NULL);
	XtVaSetValues(_widget,
		      XtNheight, (_nitems > _nitemsShown ?
				  height * _nitemsShown / _nitems : height),
		      NULL);
    }
}

void
ListCmd::setPercent(float percent)
{
    Dimension	viewportHeight, listHeight;
    XtVaGetValues(_widget, XtNheight, &viewportHeight, NULL);
    XtVaGetValues(_list, XtNheight, &listHeight, NULL);
    u_int	nshown = _nitems * viewportHeight / listHeight;
    _top = int(_nitems * percent + 0.5);
    if (_top < 0)
	_top = 0;
    else if (_top > _nitems - nshown)
	_top = _nitems - nshown;
    vViewportSetLocation(_widget, 0.0, (float)_top / (float)_nitems);
}

void
ListCmd::scroll(int n)
{
    setPercent((float)(_top + n) / (float)_nitems);
}

}
}
