/*
 *  $Id: Pata.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
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
