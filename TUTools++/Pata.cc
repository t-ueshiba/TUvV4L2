/*
 *  $Id: Pata.cc,v 1.3 2003-07-06 23:53:21 ueshiba Exp $
 */
#ifndef __APPLE__

#include "TU/Serial++.h"

namespace TU
{
/************************************************************************
*  class Pata								*
************************************************************************/
Pata::Pata(const char* ttyname)
    :Serial(ttyname)
{
    o_through().i_through();
}
 
}
#endif	/* !__APPLE__	*/
