/*
 *  $Id$
 */
#include "MyCanvasPane.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyCanvasPane							*
************************************************************************/
void
MyCanvasPane::repaintUnderlay()
{
    _dc << _image;
}

void
MyCanvasPane::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case Id_MouseButton1Press:
      case Id_MouseButton1Drag:
      case Id_MouseButton1Release:
      case Id_MouseMove:
      {
	CmdVal	logicalPosition(_dc.dev2logU(val.u), _dc.dev2logV(val.v));
	parent().callback(id, logicalPosition);
      }
        return;
    }

    parent().callback(id, val);
}
    
}
}
