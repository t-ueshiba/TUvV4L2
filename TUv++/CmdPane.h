/*
 *  $Id: CmdPane.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvCmdPane_h
#define __TUvCmdPane_h

#include "TU/v/CmdWindow.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class CmdPane							*
************************************************************************/
class CmdPane : public Pane, public CmdParent
{
  public:
    CmdPane(Window& parentWindow, const CmdDef cmd[])		;
    virtual ~CmdPane()						;

    virtual const Widget&	widget()		const	;

  private:
    const Widget	_widget;		// boxWidget
};

}
}
#endif	// !__TUvCmdPane_h
