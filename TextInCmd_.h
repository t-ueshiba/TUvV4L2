/*
 *  $Id: TextInCmd_.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvTextInCmd_h
#define __TUvTextInCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class TextInCmd							*
************************************************************************/
class TextInCmd : public Cmd
{
  public:
    TextInCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~TextInCmd()					;

    virtual const Widget&	widget()		const	;
  
    virtual const char*	getString()			const	;
    virtual void	setString(const char* str)		;
    
  private:
    const Widget	_widget;			// asciiTextWidget
};

}
}
#endif	// !__TUvTextInCmd_h
