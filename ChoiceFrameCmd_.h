/*
 *  $Id: ChoiceFrameCmd_.h,v 1.2 2002-07-25 02:38:10 ueshiba Exp $
 */
#ifndef __TUvChoiceFrameCmd_h
#define __TUvChoiceFrameCmd_h

#include "FrameCmd_.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ChoiceFrameCmd							*
************************************************************************/
class ChoiceFrameCmd : public FrameCmd
{
  public:
    ChoiceFrameCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual		 ~ChoiceFrameCmd()				;

    virtual void	callback(CmdId id, CmdVal val)			;
    virtual CmdVal	getValue()				const	;
    virtual void	setValue(CmdVal val)				;
};

}
}
#endif	// !__TUvChoiceFrameCmd_h
