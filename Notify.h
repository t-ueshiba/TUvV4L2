/*
 *  $Id: Notify.h,v 1.2 2002-07-25 02:38:12 ueshiba Exp $
 */
#ifndef __TUvNotify_h
#define __TUvNotify_h

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
#endif // !__TUvNotify_h
