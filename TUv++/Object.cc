/*
 *  $Id: Object.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class Object								*
************************************************************************/
void
Object::callback(CmdId id, CmdVal val)
{
    parent().callback(id, val);
}

void
Object::tick()
{
}

/*
 *  protected member functions
 */
Object::~Object()
{
}

App&
Object::app()
{
    return _parent.app();
}

Window&
Object::window()
{
    return _parent.window();
}

CanvasPane&
Object::canvasPane()
{
    return _parent.canvasPane();
}

}
}
