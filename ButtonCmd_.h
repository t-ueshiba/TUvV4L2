/*
 *  $Id: ButtonCmd_.h,v 1.2 2002-07-25 02:38:09 ueshiba Exp $
 */
#ifndef __TUvButtonCmd_h
#define __TUvButtonCmd_h

#include "TU/v/TUv++.h"
#include "TU/v/Bitmap.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ButtonCmd							*
************************************************************************/
class ButtonCmd : public Cmd
{
  public:
    ButtonCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~ButtonCmd()					;
    
    virtual const Widget&	widget()		const	;

    
  private:
    const Widget	_widget;			// commandWidget
    Bitmap* const	_bitmap;
};

}
}
#endif	// !__TUvButtonCmd_h
