/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: TriggerGenerator.cc,v 1.9 2007-11-26 07:55:48 ueshiba Exp $
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
    using namespace	std;

    *this >> through << through
	  << baud(9600) << csize(8) << noparity << stop1;
    setf(ios::uppercase);
}

TriggerGenerator&
TriggerGenerator::showId(std::ostream& o)
{
    using namespace	std;
    
    *this << 'V' << endl;
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
    *this << 'I' << std::endl;
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

