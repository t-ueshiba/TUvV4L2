/*
 *  $Id$  
 */
#ifndef TU_V_MODALDIALOG_H
#define TU_V_MODALDIALOG_H

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
#endif	// !TU_V_MODALDIALOG_H
