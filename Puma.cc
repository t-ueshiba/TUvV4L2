/*
 *  $Id: Puma.cc,v 1.2 2002-07-25 02:38:06 ueshiba Exp $
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
    puma.std::ostream::operator <<(c);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, const char* s)
{
    puma.std::ostream::operator <<(s);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, int i)
{
    puma.std::ostream::operator <<(i);
    return puma;
}

static inline Puma&
operator <<(Puma& puma, float f)
{
    puma.std::ostream::operator <<(f);
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
operator <<(Puma& puma, const Vector<float>& position)
{
    if (position.dim() > 0)
    {
	puma << "poi tmp" << endc;		// wait for "Change ? "
	const u_int	n = (position.dim() > 6 ? 6 : position.dim()) - 1;
	for (u_int i = 0; i < n; i++)
	    puma << position[i] << ',';
	puma << position[n] << endc;		// wait for "Change ? "
	puma << endc;				// wait for prompt
	puma << "do move tmp" << endc;		// wait for prompt
    }
    return puma;
}

Puma&
operator >>(Puma& puma, Vector<float>& position)
{
    puma << "where\r" >> skipl >> skipl;	// ignore "X Y Z O A T"
    operator >>((std::istream&)puma, position);
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
