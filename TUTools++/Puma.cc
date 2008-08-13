/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: Puma.cc,v 1.11 2008-08-13 06:05:34 ueshiba Exp $
 */
#include "TU/Serial++.h"

namespace TU
{
/************************************************************************
*  Static functions							*
************************************************************************/ 
static inline Puma&
operator <<(Puma& puma, char c)
{
    puma.Serial::operator <<(c);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, const char* s)
{
    puma.Serial::operator <<(s);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, int i)
{
    puma.Serial::operator <<(i);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, float f)
{
    puma.Serial::operator <<(f);
    return puma;
}

static Puma&
endc(Puma& puma)
{
    return puma << '\r' << wait;
}

static int
match_msg(const char* p[], int nmsg)
{
    for (int i = 0; i < nmsg; i++)
	if (*p[i] == '\0')
	    return i;
    return -1;
}

/************************************************************************
*  class Puma								*
************************************************************************/
Puma::Puma(const char* ttyname)
    :Serial(ttyname), _axis(Jt1), _echo(NoEcho)
{
    o_through().i_igncr();
    *this << '\r';
    if (wait() == 0)		/* wait for "... floppy (Y/N)? "	*/
	return;
    *this << "n\r";
    if (wait() == 0)		/* wait for "Initialize (Y/N)? "	*/
	return;
    *this << 'n' << endc;	/* wait for prompt			*/
}

Puma&
Puma::operator +=(int dangle)
{
    return *this << "do drive " << (int)_axis << ',' << dangle << ",100"
		 << endc;
}

Puma&
Puma::operator -=(int dangle)
{
    return *this << "do drive " << (int)_axis << ',' << -dangle << ",100"
		 << endc;
}

int
Puma::wait()
{
    static const char* const	msg[] = {"\n.", "? ", "?\n"};
    static const int		nmsg = sizeof(msg) / sizeof(msg[0]);
    static const char*		p[nmsg];
    int				i;

    for (i = 0; i < nmsg; i++)
	p[i] = msg[i];
    
    while ((i = match_msg(p, nmsg)) == -1)
    {
	char	c;
	
	if (!get(c))
	    break;
	else if (c == '\0')
	    continue;
	
	if (_echo)
	    std::cerr << c;

	for (i = 0; i < nmsg; i++)
	    if (c == *p[i] || c == *(p[i] = msg[i]))
		++p[i];
    }
    
    return i;
}

Puma&
Puma::set_axis(Puma::Axis axis)
{
    _axis = axis;
    return *this;
}

/************************************************************************
*  friends of class Puma						*
************************************************************************/
Puma&
operator <<(Puma& puma, const Puma::Position& position)
{
    puma << "poi tmp" << endc;		// wait for "Change ? "
    for (u_int i = 0; i < 5; i++)
	puma << position[i] << ',';
    puma << position[5] << endc;		// wait for "Change ? "
    puma << endc;				// wait for prompt
    puma << "do move tmp" << endc;		// wait for prompt

    return puma;
}

Puma&
operator >>(Puma& puma, Puma::Position& position)
{
    puma << "where\r" >> skipl >> skipl;	// ignore "X Y Z O A T"
    operator >>((Serial&)puma, position);
    return puma << wait;			// wait for prompt
}

/*
 *  Manipulators
 */
OManip1<Puma, Puma::Axis>
axis(Puma::Axis axis)
{
    return OManip1<Puma, Puma::Axis>(&Puma::set_axis, axis);
}


Puma&	wait   (Puma& puma)	{puma.wait(); return puma;}
Puma&	calib  (Puma& puma)	{return puma << "cal" << endc << 'y' << endc;}
Puma&	ready  (Puma& puma)	{return puma << "do ready" << endc;}
Puma&	nest   (Puma& puma)	{return puma << "do nest" << endc;}
Puma&	echo   (Puma& puma)	{puma._echo = Puma::DoEcho; return puma;}
Puma&	no_echo(Puma& puma)	{puma._echo = Puma::NoEcho; return puma;}

}

