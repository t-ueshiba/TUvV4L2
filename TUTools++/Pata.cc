/*
 *  $Id: Pata.cc,v 1.2 2002-07-25 02:38:06 ueshiba Exp $
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
