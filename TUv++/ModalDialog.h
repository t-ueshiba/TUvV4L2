/*
 *  $Id: ModalDialog.h,v 1.2 2002-07-25 02:38:12 ueshiba Exp $
 */
#ifndef __TUvModalDialog_h
#define __TUvModalDialog_h

#include "TU/v/Dialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ModalDialog							*
************************************************************************/
class ModalDialog : public Dialog
{
  public:
    ModalDialog(Window& parentWindow, const char* myName, 
		const CmdDef cmd[])				;
    virtual ~ModalDialog()					;

    virtual void	show()					;
    virtual void	hide()					;
    
  private:
    bool		_active;
};

}
}
#endif	// !__TUvModalDialog_h
