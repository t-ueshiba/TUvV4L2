/*
 *  $Id: Serial++.h,v 1.3 2002-07-25 11:53:22 ueshiba Exp $
 */
#ifndef __TUSerialPP_h
#define __TUSerialPP_h

#include <termios.h>
#ifndef sgi
#  include <fstream>
#else
#  include <fstream.h>
#endif
#include "TU/Manip.h"
#include "TU/Vector++.h"

namespace TU
{
/************************************************************************
*  class Serial								*
************************************************************************/
#ifndef sgi
class Serial : public std::fstream
#else
class Serial : public fstream
#endif
{
  public:
		Serial(const char*)			;
    virtual	~Serial()				;

    Serial&	i_nl2cr()				;
    Serial&	i_igncr()				;
    Serial&	i_cr2nl()				;
#ifndef __APPLE__
    Serial&	i_upper2lower()				;
#endif
    Serial&	i_through()				;
    Serial&	o_nl2crnl()				;
#ifndef __APPLE__
    Serial&	o_cr2nl()				;
    Serial&	o_lower2upper()				;
#endif
    Serial&	o_through()				;
    Serial&	c_baud(int)				;
    Serial&	c_csize(int)				;
    Serial&	c_even()				;
    Serial&	c_odd()					;
    Serial&	c_noparity()				;
    Serial&	c_stop1()				;
    Serial&	c_stop2()				;
    
  private:
    int		fd()					{return rdbuf()->fd();}
    Serial&	set_flag(tcflag_t termios::* flag,
			 unsigned long clearbits,
			 unsigned long setbits)		;

    termios	_termios_bak;
};

inline Serial&
operator >>(Serial& serial, Serial& (*f)(Serial&))
{
    return f(serial);
}

inline Serial&
operator <<(Serial& serial, Serial& (*f)(Serial&))
{
    return f(serial);
}

#ifdef sgi
::istream&	ign(::istream& in)	;
::istream&	skipl(::istream& in)	;
#endif
extern IOManip<Serial>	nl2cr;
#ifndef __APPLE__
extern IOManip<Serial>	cr2nl;
extern IOManip<Serial>	upperlower;
#endif
extern IOManip<Serial>	through;
Serial&			igncr	(Serial&)		;
Serial&			even	(Serial&)		;
Serial&			odd	(Serial&)		;
Serial&			noparity(Serial&)		;
Serial&			stop1	(Serial&)		;
Serial&			stop2	(Serial&)		;
OManip1<Serial, int>	baud	(int)			;
OManip1<Serial, int>	csize	(int)			;

/************************************************************************
*  class Puma								*
************************************************************************/
class Puma : public Serial
{
  public:
    enum Axis		{Jt1=1, Jt2=2, Jt3=3, Jt4=4, Jt5=5, Jt6=6};
    
			Puma(const char*)			;
    
    Puma&		operator +=(int)			;
    Puma&		operator -=(int)			;
    
    friend Puma&	wait(Puma&)				;
    friend Puma&	echo(Puma&)				;
    friend Puma&	no_echo(Puma&)				;
    friend OManip1<Puma, Axis>	axis(Puma::Axis)		;
    
  private:
    enum Echo		{NoEcho = 0, DoEcho = 1};

    Puma&		set_axis(Puma::Axis)			;
    int			wait()					;

    Axis		_axis;
    Echo		_echo;
};

inline Puma&
operator <<(Puma& puma, Puma& (*f)(Puma&))
{
    return (*f)(puma);
}

Puma&	operator <<(Puma&, const Vector<float>&)		;
Puma&	operator >>(Puma&, Vector<float>&)			;
Puma&	calib(Puma&)						;
Puma&	ready(Puma&)						;
Puma&	nest (Puma&)						;

/************************************************************************
*  class Pata								*
************************************************************************/ 
class Pata : public Serial
{
  public:
		Pata(const char*)				;

  private:
    enum	{SX=2, EX=3};
};

/************************************************************************
*  class Microscope							*
************************************************************************/ 
class Microscope : public Serial
{
  public:
    enum Axis	{X = 'X', Y = 'Y', Z = 'Z'};

		Microscope(const char*)				;

    Microscope&	operator +=(int)				;
    Microscope&	operator -=(int)				;
    Microscope&	operator ++()					;
    Microscope&	operator --()					;
    
    friend OManip1<Microscope, Microscope::Axis>
		axis(Microscope::Axis)				;
    
  private:
    Microscope&	set_axis(Microscope::Axis)			;
    
    Axis	_axis;
};

inline Microscope&
operator <<(Microscope& ms, Microscope& (*f)(Microscope&))
{
    return (*f)(ms);
}

Microscope&	operator <<(Microscope&, const Array<int>&)	;
Microscope&	operator >>(Microscope&, Array<int>&)		;
Microscope&	calib(Microscope&)				;
Microscope&	ready(Microscope&)				;

/************************************************************************
*  class TriggerGenerator						*
************************************************************************/
class TriggerGenerator : public Serial
{
  public:
    TriggerGenerator(const char* ttyname)			;

    TriggerGenerator&	showId(std::ostream& out)		;
    TriggerGenerator&	selectChannel(u_int channel)		;
    TriggerGenerator&	setInterval(u_int interval)		;
    TriggerGenerator&	oneShot()				;
    TriggerGenerator&	continuousShot()			;
    TriggerGenerator&	stopContinuousShot()			;
    int			getConfiguration(u_int& channel,
					 u_int& interval)	;
};
 
}

#endif	/* !__TUSerialPP_h	*/
