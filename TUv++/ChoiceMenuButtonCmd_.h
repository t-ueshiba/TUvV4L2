/*
 *  $Id: ChoiceMenuButtonCmd_.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvChoiceMenuButtonCmd_h
#define __TUvChoiceMenuButtonCmd_h

#include "TU/v/Menu.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ChoiceMenuButtonCmd						*
************************************************************************/
class ChoiceMenuButtonCmd : public Cmd
{
  public:
    ChoiceMenuButtonCmd(Object& parentObject, const CmdDef& cmd)	;
    virtual ~ChoiceMenuButtonCmd()					;
    
    virtual const Widget&	widget()			const	;
    
    virtual void	callback(CmdId id, CmdVal val)			;
    virtual CmdVal	getValue()				const	;
    virtual void	setValue(CmdVal val)				;

  private:
    const Widget	_widget;			// menuButtonWidget
    ChoiceMenu		_menu;
};

}
}
#endif	// !__TUvChoiceMenuButtonCmd_h
