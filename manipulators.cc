/*
 *  $Id: manipulators.cc,v 1.1.1.1 2002-07-25 02:14:16 ueshiba Exp $
 */
#include <iostream>

namespace TU
{
/************************************************************************
*  Manipulators for std::istream					*
************************************************************************/
std::istream&
ign(std::istream& in)	// manipulator for skipping the rest of a line
{
    char	c;
    while (in.get(c))
	if (c == '\n')
	    break;
    return in;
}

std::istream&
skipl(std::istream& in)
{
    char	c;
    
    while (in.get(c))
	if (c == '\n' || c == '\r')
	    break;
    return in;
}
 
}
