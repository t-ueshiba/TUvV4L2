/*
 *  $Id$  
 */
#ifndef TU_V_CONFIRM_H
#define TU_V_CONFIRM_H

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
#endif	// !TU_V_CONFIRM_H
