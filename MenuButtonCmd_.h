/*
 *  $Id: MenuButtonCmd_.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvMenuButtonCmd_h
#define __TUvMenuButtonCmd_h

#include "TU/v/Menu.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MenuButtonCmd							*
************************************************************************/
class MenuButtonCmd : public Cmd
{
  public:
    MenuButtonCmd(Object& parentObject, const CmdDef& cmd)	;
    virtual ~MenuButtonCmd()					;
    
    virtual const Widget&	widget()		const	;
    
  private:
    const Widget	_widget;			// menuButtonWidget

  protected:
    Menu		_menu;
};

}
}
#endif	// !__TUvMenuButtonCmd_h
