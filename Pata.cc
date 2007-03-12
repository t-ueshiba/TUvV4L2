/*
 *  $Id: Pata.cc,v 1.5 2007-03-12 07:15:29 ueshiba Exp $
 */
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

