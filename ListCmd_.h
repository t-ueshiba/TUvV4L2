/*
 *  $Id: ListCmd_.h,v 1.2 2002-07-25 02:38:11 ueshiba Exp $
 */
#ifndef __TUvListCmd_h
#define __TUvListCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ListCmd							*
************************************************************************/
class ListCmd : public Cmd
{
  public:
    ListCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~ListCmd()						;

    virtual const Widget&	widget()		const	;

    virtual CmdVal		getValue()		const	;
    virtual void		setValue(CmdVal val)		;
    virtual void		setProp(void* prop)		;
    void			setPercent(float percent)	;
    void			scroll(int n)			;
    
  private:
    const Widget	_widget;			// viewportWidget
    const Widget	_list;				// listWidget
    int			_top;
    u_int		_nitems;
    const u_int		_nitemsShown;
};

}
}
#endif	// !__TUvListCmd_h
