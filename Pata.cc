/*
 *  $Id: Pata.cc,v 1.4 2004-04-28 02:28:28 ueshiba Exp $
 */
#if (!defined(__GNUC__) || (__GNUC__ < 3))

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
#endif
