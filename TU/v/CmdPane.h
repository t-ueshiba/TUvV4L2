/*
 *  $Id$  
 */
#ifndef TU_V_CMDPANE_H
#define TU_V_CMDPANE_H

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
#endif	// !TU_V_CMDPANE_H
