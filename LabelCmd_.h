/*
 *  $Id: LabelCmd_.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvLabelCmd_h
#define __TUvLabelCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class LabelCmd							*
************************************************************************/
class LabelCmd : public Cmd
{
  public:
    LabelCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~LabelCmd()						;

    virtual const Widget&	widget()		const	;
    
  private:
    const Widget	_widget;	// labelWidget
};

}
}
#endif	// !__TUvLabelCmd_h
