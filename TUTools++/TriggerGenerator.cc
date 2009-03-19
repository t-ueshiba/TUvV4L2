/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: TriggerGenerator.cc,v 1.14 2009-03-19 05:11:03 ueshiba Exp $
 */
#include <cstdlib>
#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static void
skipl(FILE* fp)
{
    for (int c; (c = fgetc(fp)) != EOF; )
	if (c == '\n')
	    break;
}
    
/************************************************************************
*  class TriggerGenerator						*
************************************************************************/
TriggerGenerator::TriggerGenerator(const char* ttyname)
    :Serial(ttyname)
{
    i_through()
	.o_through().o_lower2upper()
	.c_baud(9600).c_csize(8).c_noparity().c_stop1();
}

TriggerGenerator&
TriggerGenerator::showId(std::ostream& out)
{
    using namespace	std;
    
    fputs("V\n", fp());
    for (int c; (c = fgetc(fp())) != EOF; )
    {
	out << char(c);
	if (c == '\n')
	    break;
    }
    return *this;
}

TriggerGenerator&
TriggerGenerator::selectChannel(u_int channel)
{
    fprintf(fp(), "A%0.8x\n", channel);
    skipl(fp());
    return *this;
}

TriggerGenerator&
TriggerGenerator::setInterval(u_int interval)
{
    if (10 <= interval && interval <= 255)
    {
	fprintf(fp(), "F%d\n", interval);
	skipl(fp());
    }
    return *this;
}

TriggerGenerator&
TriggerGenerator::oneShot()
{
    fputs("T\n", fp());
    skipl(fp());
    return *this;
}

TriggerGenerator&
TriggerGenerator::continuousShot()
{
    fputs("R\n", fp());
    skipl(fp());
    return *this;
}

TriggerGenerator&
TriggerGenerator::stopContinuousShot()
{
    fputs("S\n", fp());
    skipl(fp());
    return *this;
}

bool
TriggerGenerator::getStatus(u_int& channel, u_int& interval) const
{
    fputs("I\n", fp());

    char	token[64], *p = token;
    for (int c; (c = fgetc(fp())) != EOF; )
    {
	if (c == '\n')
	    break;
	else if (c == ',')
	{
	    *p = '\0';
	    if (token[0] == 'A')
		channel = strtoul(token + 1, NULL, 16);
	    else
		interval = strtoul(token + 1, NULL, 10);
	    p = token;
	}
	else
	    *p++ = c;
    }
    *p = '\0';
    
    return !strcmp(token, "RUN");
}

}

