/*
 *  $Id$  
 */
#ifndef TU_V_NOTIFY_H
#define TU_V_NOTIFY_H

#include <sstream>
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Notify								*
************************************************************************/
class Notify : public ModalDialog, public std::ostringstream
{
  public:
    Notify(Window& parentWindow)					;
    virtual		~Notify()					;

    virtual void	show()						;
    
    virtual void	callback(CmdId id, CmdVal val)			;
};

}
}
#endif // !TU_V_NOTIFY_H
