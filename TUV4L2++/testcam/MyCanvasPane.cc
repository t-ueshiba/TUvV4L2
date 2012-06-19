/*
 *  $Id: MyCanvasPane.cc,v 1.1 2012-06-19 06:14:31 ueshiba Exp $
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
