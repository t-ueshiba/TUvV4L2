/*
 *  $Id: Serial.cc,v 1.7 2007-03-12 07:43:51 ueshiba Exp $
 */
#include "TU/Serial++.h"
#include <stdexcept>
#include <string>
#include <errno.h>
#ifdef HAVE_STDIO_FILEBUF
#  include <fcntl.h>
#endif

namespace TU
{
/************************************************************************
*  Public member functions						*
************************************************************************/ 
Serial::Serial(const char* ttyname)
#ifdef HAVE_STDIO_FILEBUF
    :std::basic_iostream<char>(NULL),
#  if (__GNUC__ < 4)
     _filebuf(::open(ttyname, O_RDWR, S_IRUSR | S_IWUSR),
	      ios_base::in|ios_base::out, true, BUFSIZ)
#  else
     _filebuf(::open(ttyname, O_RDWR, S_IRUSR | S_IWUSR),
	      ios_base::in|ios_base::out)
#  endif
#else
    :std::fstream(ttyname, ios_base::in|ios_base::out)
#endif
{
#ifdef HAVE_STDIO_FILEBUF
    init(&_filebuf);
    if (fd() < 0)
	setstate(ios_base::failbit);
    else
	clear();
#endif
    if (!*this)
	throw std::runtime_error(std::string("TU::Serial::Serial: cannot open fstream; ")
				 + strerror(errno));
    
    termios	termios;
    if (::tcgetattr(fd(), &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw std::runtime_error(std::string("TU::Serial::Serial: tcgetattr; ")
				 + strerror(errno));
    }

    _termios_bak = termios;		// backup termios structure
    termios.c_lflag &= ~(ICANON | ECHO | ISIG);
    termios.c_cc[VMIN]  = 1;
    termios.c_cc[VTIME] = 0;
    if (::tcsetattr(fd(), TCSANOW, &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw std::runtime_error(std::string("TU::Serial::Serial: tcsetattr; ")
				 + strerror(errno));
    }
}

Serial::~Serial()
{
    if (*this)
	tcsetattr(fd(), TCSANOW, &_termios_bak);
}

/*
 *  input flags
 */
Serial&
Serial::i_nl2cr()		// '\n' -> '\r'
{
    return set_flag(&termios::c_iflag, ICRNL, INLCR);
}

Serial&
Serial::i_igncr()		// don't read '\r'
{
    return set_flag(&termios::c_iflag, INLCR|ICRNL, IGNCR);
}

Serial&
Serial::i_cr2nl()		// '\r' -> '\n'
{
    return set_flag(&termios::c_iflag, INLCR, ICRNL);
}

#ifndef __APPLE__
Serial&
Serial::i_upper2lower()		// upper -> lower
{
    return set_flag(&termios::c_iflag, 0, IUCLC);
}
#endif

Serial&
Serial::i_through()		// read transparently
{
#ifndef __APPLE__
    return set_flag(&termios::c_iflag, INLCR|IGNCR|ICRNL|IUCLC, 0);
#else
    return set_flag(&termios::c_iflag, INLCR|IGNCR|ICRNL, 0);
#endif
}

/*
 *  output flags
 */
Serial&
Serial::o_nl2crnl()		// '\r\n' <- "\n"
{
#ifndef __APPLE__
    return set_flag(&termios::c_oflag, OCRNL, OPOST|ONLCR);
#else
    return set_flag(&termios::c_oflag, 0, OPOST|ONLCR);
#endif
}

#ifndef __APPLE__
Serial&
Serial::o_cr2nl()		// '\n' <- '\r'
{
    return set_flag(&termios::c_oflag, ONLCR, OPOST|OCRNL);
}

Serial&
Serial::o_lower2upper()	// upper <- lower
{
    return set_flag(&termios::c_oflag, 0, OPOST|OLCUC);
}
#endif

Serial&
Serial::o_through()		// write transparently
{
    return set_flag(&termios::c_oflag, OPOST, 0);
}

/*
 *  control flags
 */
Serial&
Serial::c_baud(int baud)	// set baud rate
{
#ifndef __APPLE__
    switch (baud)
    {
      case 50:
	return set_flag(&termios::c_cflag, CBAUD, B50);
      case 75:
	return set_flag(&termios::c_cflag, CBAUD, B75);
      case 110:
	return set_flag(&termios::c_cflag, CBAUD, B110);
      case 134:
	return set_flag(&termios::c_cflag, CBAUD, B134);
      case 150:
	return set_flag(&termios::c_cflag, CBAUD, B150);
      case 200:
	return set_flag(&termios::c_cflag, CBAUD, B200);
      case 300:
	return set_flag(&termios::c_cflag, CBAUD, B300);
      case 600:
	return set_flag(&termios::c_cflag, CBAUD, B600);
      case 1200:
	return set_flag(&termios::c_cflag, CBAUD, B1200);
      case 1800:
	return set_flag(&termios::c_cflag, CBAUD, B1800);
      case 2400:
	return set_flag(&termios::c_cflag, CBAUD, B2400);
      case 4800:
	return set_flag(&termios::c_cflag, CBAUD, B4800);
      case 9600:
	return set_flag(&termios::c_cflag, CBAUD, B9600);
      case 19200:
	return set_flag(&termios::c_cflag, CBAUD, B19200);
      case 38400:
	return set_flag(&termios::c_cflag, CBAUD, B38400);
    }
#else
    termios		termios;

    if (::tcgetattr(fd(), &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw std::runtime_error(std::string("TU::Serial::c_baud: tcgetattr; ")
				 + strerror(errno));
    }
    termios.c_ispeed = termios.c_ospeed = baud;
    if (::tcsetattr(fd(), TCSANOW, &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw std::runtime_error(std::string("TU::Serial::c_baud: tcsetattr; ")
				 + strerror(errno));
    }
#endif
    return *this;
}

Serial&
Serial::c_csize(int csize)	// set character size
{
    switch (csize)
    {
      case 5:
	return set_flag(&termios::c_cflag, CSIZE, CS5);
      case 6:
	return set_flag(&termios::c_cflag, CSIZE, CS6);
      case 7:
	return set_flag(&termios::c_cflag, CSIZE, CS7);
      case 8:
	return set_flag(&termios::c_cflag, CSIZE, CS8);
    }
    return *this;
}

Serial&
Serial::c_even()		// even parity
{
    return set_flag(&termios::c_cflag, PARODD, PARENB);
}

Serial&
Serial::c_odd()			// odd parity
{
    return set_flag(&termios::c_cflag, 0, PARENB|PARODD);
}

Serial&
Serial::c_noparity()		// no parity
{
    return set_flag(&termios::c_cflag, PARENB, 0);
}

Serial&
Serial::c_stop1()		// 1 stop bit
{
    return set_flag(&termios::c_cflag, CSTOPB, 0);
}

Serial&
Serial::c_stop2()		// 2 stop bits
{
    return set_flag(&termios::c_cflag, 0, CSTOPB);
}

/************************************************************************
*  Private member functions						*
************************************************************************/ 
Serial&
Serial::set_flag(tcflag_t termios::* flag,
		 unsigned long clearbits, unsigned long setbits)
{
    termios		termios;

    if (::tcgetattr(fd(), &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw
	    std::runtime_error(std::string("TU::Serial::set_flag: tcgetattr; ")
			       + strerror(errno));
    }
    termios.*flag &= ~clearbits;
    termios.*flag |= setbits;
    if (::tcsetattr(fd(), TCSANOW, &termios) == -1)
    {
	clear(ios_base::badbit|rdstate());
	throw
	    std::runtime_error(std::string("TU::Serial::set_flag: tcsetattr; ")
			       + strerror(errno));
    }
    return *this;
}

/************************************************************************
*  Manipulators for Serial						*
************************************************************************/
Serial&	igncr	(Serial& s)	{return s.i_igncr();}
Serial&	even	(Serial& s)	{return s.c_even();}
Serial&	odd	(Serial& s)	{return s.c_odd();}
Serial&	noparity(Serial& s)	{return s.c_noparity();}
Serial&	stop1	(Serial& s)	{return s.c_stop1();}
Serial&	stop2	(Serial& s)	{return s.c_stop2();}

OManip1<Serial, int>
baud(int bps)
{
    return OManip1<Serial, int>(&Serial::c_baud, bps);
}

OManip1<Serial, int>
csize(int cs)
{
    return OManip1<Serial, int>(&Serial::c_csize, cs);
}
 
}

