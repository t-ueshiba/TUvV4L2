/*
 *  $Id: XilObject.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/XilDC.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class XilObject							*
************************************************************************/
XilObject::XilObject()
{
    if (_nobjects++ == 0)
	if ((_xilstate = xil_open()) == NULL)
	    throw std::runtime_error("Failed to open XIL!!");
}

XilObject::~XilObject()
{
    if (_nobjects > 0 && --_nobjects == 0)
    {
      //	xil_close(_xilstate);
	_xilstate = NULL;
    }
}
 
}
