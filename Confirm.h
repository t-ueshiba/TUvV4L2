/*
 *  $Id: Confirm.h,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#ifndef __TUvConfirm_h
#define __TUvConfirm_h

#include <sstream>
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Confirm							*
************************************************************************/
class Confirm : public ModalDialog, public std::ostringstream
{
  public:
    Confirm(Window& parentWindow)					;
    virtual		~Confirm()					;

    bool		ok()						;
    
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    bool		_ok;
};

}
}
#endif	// !__TUvConfirm_h
