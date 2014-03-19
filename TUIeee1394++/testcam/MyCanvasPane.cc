/*
 *  $Id: MyCanvasPane.cc,v 1.2 2010-11-19 06:31:08 ueshiba Exp $
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

}
}
