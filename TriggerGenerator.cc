/*
 *  $Id: TriggerGenerator.cc,v 1.2 2002-07-25 02:38:07 ueshiba Exp $
 */
#include <iomanip>
#include <cstdlib>
#include "TU/Serial++.h"

namespace TU
{
/************************************************************************
*  class TriggerGenerator						*
************************************************************************/
TriggerGenerator::TriggerGenerator(const char* ttyname)
    :Serial(ttyname)
{
    *this >> through << through
	  << baud(9600) << csize(8) << noparity << stop1;
    setf(std::ios::uppercase);
}

TriggerGenerator&
TriggerGenerator::showId(std::ostream& o)
{
    *this << 'V' << std::endl;
    for (char c; get(c); )
    {
	o << c;
	if (c == '\n')
	    break;
    }
    return *this;
}

TriggerGenerator&
TriggerGenerator::selectChannel(u_int channel)
{
    using namespace	std;
    
    setf(ios::hex, ios::basefield);
    *this << 'A';
    width(8);
    fill('0');
    *this << channel << endl;
    *this >> ign;
    return *this;
}

TriggerGenerator&
TriggerGenerator::setInterval(u_int interval)
{
    using namespace	std;
    
    if (10 <= interval && interval <= 255)
    {
	setf(ios::dec, ios::basefield);
	*this << 'F' << interval << endl;
	*this >> ign;
    }
    return *this;
}

TriggerGenerator&
TriggerGenerator::oneShot()
{
    *this << 'T' << std::endl;
    *this >> ign;
    return *this;
}

TriggerGenerator&
TriggerGenerator::continuousShot()
{
    *this << 'R' << std::endl;
    *this >> ign;
    return *this;
}

TriggerGenerator&
TriggerGenerator::stopContinuousShot()
{
    *this << 'S' << std::endl;
    *this >> ign;
    return *this;
}

int
TriggerGenerator::getConfiguration(u_int& channel, u_int& interval)
{
    using namespace	std;
    
    *this << 'I' << endl;
    char	token[64];
    for (char c, *p = token; get(c); )
    {
	if (c == '\n')
	{
	    *p = '\0';
	    break;
	}
	else if (c == ',')
	{
	    *p = '\0';
	    p = token;
	    if (token[0] == 'A')
		channel = strtoul(token + 1, NULL, 16);
	    else
		interval = strtoul(token + 1, NULL, 10);
	}
	else
	    *p++ = c;
    }
    
    return !strcmp(token, "RUN");
}

}
