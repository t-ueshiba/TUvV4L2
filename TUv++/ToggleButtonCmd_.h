/*
 *  $Id: ToggleButtonCmd_.h,v 1.2 2002-07-25 02:38:13 ueshiba Exp $
 */
#ifndef __TUvToggleButtonCmd_h
#define __TUvToggleButtonCmd_h

#include "TU/v/TUv++.h"
#include "TU/v/Bitmap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ToggleButtonCmd						*
************************************************************************/
class ToggleButtonCmd : public Cmd
{
  public:
    ToggleButtonCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~ToggleButtonCmd()						;
    
    virtual const Widget&	widget()			const	;

    virtual CmdVal		getValue()			const	;
    virtual void		setValue(CmdVal val)			;
    
  private:
    const Widget	_widget;			// toggleWidget
    Bitmap* const	_bitmap;
};

}
}
#endif	// !__TUvToggleButtonCmd_h
