/*
 *  $Id: Menu.h,v 1.1.1.1 2002-07-25 02:14:18 ueshiba Exp $
 */
#ifndef __TUvMenu_h
#define __TUvMenu_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Menu							*
************************************************************************/
class Menu : public Cmd
{
  public:
    class Item : public Cmd
    {
      public:
	Item(Menu& parentMenu, const MenuDef& menuItem)			;
	virtual			~Item()					;
	
	virtual const Widget&	widget()			const	;

	virtual void	callback(CmdId id, CmdVal val)			;
	virtual CmdVal	getValue()				const	;
	virtual void	setValue(CmdVal val)				;

      private:
	const Widget	_widget;	// smeLineObject or smeBSBObject

	static u_int	_nitems;
    };

  public:
    Menu(Object& parentObject, const MenuDef menu[])			;
    Menu(Object& parentObject, const MenuDef menu[],
	 const char* name, ::Widget parentWidget)			;
    virtual			~Menu()					;
	    
    virtual const Widget&	widget()			const	;

  private:
    const Widget		_widget;	// simpleMenuWidget
};

/************************************************************************
*  class ChoiceMenu							*
************************************************************************/
class ChoiceMenu : public Menu
{
  public:
    ChoiceMenu(Object& parentObject, const MenuDef menu[])		;
    ChoiceMenu(Object& parentObject, const MenuDef menu[],
	       const char* name, ::Widget widget)			;
    virtual		~ChoiceMenu()					;
	    
    virtual void	callback(CmdId id, CmdVal val)			;
    virtual CmdVal	getValue()				const	;
    virtual void	setValue(CmdVal val)				;
};

}
}
#endif	// !__TUvMenu_h
