/*
 *  $Id: XglObject.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include "TU/v/XglDC.h"
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class XglObject							*
************************************************************************/
XglObject::XglObject()
{
    if (_nobjects++ == 0)
	if ((_xglstate = xgl_open(XGL_SYS_ST_ERROR_DETECTION, FALSE, NULL)) ==
	    NULL)
	    throw std::runtime_error("Failed to open XGL!!");
}

XglObject::~XglObject()
{
    if (_nobjects > 0 && --_nobjects == 0)
    {
	xgl_close(_xglstate);
	_xglstate = NULL;
    }
}
 
}
