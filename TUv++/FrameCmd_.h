/*
 *  $Id: FrameCmd_.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvFrameCmd_h
#define __TUvFrameCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class FrameCmd							*
************************************************************************/
class FrameCmd : public Cmd
{
  public:
    FrameCmd(Object& parentObject, const CmdDef& cmd)	;
    virtual ~FrameCmd()					;
    
    virtual const Widget&	widget()	const	;

  private:
    const Widget		_widget;		// frameWidget
};

}
}
#endif	// !__TUvFrameCmd_h
