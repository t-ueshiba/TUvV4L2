/*
 *  $Id: RadioButtonCmd_.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvRadioButtonCmd_h
#define __TUvRadioButtonCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class RadioButtonCmd							*
************************************************************************/
class RadioButtonCmd : public Cmd
{
  public:
    RadioButtonCmd(Object& parentObject, const CmdDef& cmd)	;
    virtual ~RadioButtonCmd()					;
    
    virtual const Widget&	widget()		const	;

    virtual CmdVal		getValue()		const	;
    virtual void		setValue(CmdVal val)		;
    
  private:
    const Widget	_widget;			// gridboxWidget
    const Widget	_button;			// commandWidget
    const Widget	_label;				// labelWidget

    static u_int	_nitems;
};

}
}
#endif	// !__TUvRadioButtonCmd_h
