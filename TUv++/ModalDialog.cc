/*
 *  $Id: ModalDialog.cc,v 1.2 2002-07-25 02:38:12 ueshiba Exp $
 */
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ModalDialog							*
************************************************************************/
ModalDialog::ModalDialog(Window& parentWindow, const char* myName,
			 const CmdDef cmd[])
    :Dialog(parentWindow, myName, cmd), _active(false)
{
}

ModalDialog::~ModalDialog()
{
}

void
ModalDialog::show()
{
    Point2<int>	p = parent().widget().position();
    p[0] += 10;
    p[1] += 10;
    XtVaSetValues(widget(), XtNx, p[0], XtNy, p[1], NULL);
    XtPopup(widget(), XtGrabExclusive);

    XtAppContext	appContext = XtWidgetToApplicationContext(widget());
    _active = true;
    while (_active)
    {
	XEvent  event;
	
	XtAppNextEvent(appContext, &event);
	XtDispatchEvent(&event);
    }

    XtPopdown(widget());
}

void
ModalDialog::hide()
{
    _active = false;
}

}
}
