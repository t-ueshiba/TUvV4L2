/*
 *  $Id: Dialog.h,v 1.2 2002-07-25 02:38:11 ueshiba Exp $
 */
#ifndef __TUvDialog_h
#define __TUvDialog_h

#include "TU/v/CmdPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Dialog								*
************************************************************************/
class Dialog : public Window
{
  public:
    Dialog(Window& parentWindow, const char* myName,
	   const CmdDef cmd[])					;
    virtual ~Dialog()						;

    virtual const Widget&	widget()		const	;

  protected:
    CmdPane&			pane()				{return _pane;}
    
  private:
    const Widget	_widget;		// transientShellWidget
    CmdPane		_pane;
};

}
}
#endif	// !__TUvDialog_h
