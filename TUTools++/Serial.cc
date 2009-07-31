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
 *  $Id: Serial.cc,v 1.18 2009-07-31 07:04:45 ueshiba Exp $
 */
#include "TU/Serial.h"
#include <stdexcept>
#include <string>
#include <errno.h>
#include <fcntl.h>

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
static int
get_fd(const char* ttyname)
{
    using namespace	std;
    
    int	fd = ::open(ttyname, O_RDWR);
    if (fd < 0)
	throw runtime_error(string("TU::Serial::Serial: cannot open tty; ")
			    + strerror(errno));
    return fd;
}
    
/************************************************************************
*  Public member functions						*
************************************************************************/
Serial::Serial(const char* ttyname)
    :_fd(get_fd(ttyname)), _fp(::fdopen(_fd, "r+"))
{
    using namespace	std;

  // Check if I/O FILE is properly opened.
    if (_fp == NULL)
	throw runtime_error(string("TU::Serial::Serial: cannot open FILE; ")
			    + strerror(errno));

  // Flush everything in the buffer.
    if (::tcflush(_fd, TCIOFLUSH))
	throw runtime_error(string("TU::Serial::Serial: cannot flush tty; ")
			    + strerror(errno));

  // Keep the original termios settings.
    termios	termios;
    if (::tcgetattr(_fd, &termios) == -1)
	throw runtime_error(string("TU::Serial::Serial: tcgetattr; ")
			    + strerror(errno));
    _termios_bak = termios;		// backup termios structure

  // Set local modes and control characters.
    termios.c_lflag &= ~(ICANON | ECHO | ISIG);
    termios.c_cc[VMIN]  = 1;
    termios.c_cc[VTIME] = 0;
    if (::tcsetattr(_fd, TCSANOW, &termios) == -1)
	throw runtime_error(string("TU::Serial::Serial: tcsetattr; ")
			    + strerror(errno));
}

Serial::~Serial()
{
    if (_fd >= 0)
    {
	::tcsetattr(_fd, TCSANOW, &_termios_bak);
	if (_fp != NULL)
	    ::fclose(_fp);
	else
	    ::close(_fd);
    }
}

const Serial&
Serial::put(const char* s) const
{
    ::fputs(s, _fp);
    return *this;
}
    
const Serial&
Serial::get(char* s, u_int size) const
{
    ::fgets(s, size, _fp);
    return *this;
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

#if !defined(__APPLE__)
Serial&
Serial::i_upper2lower()		// upper -> lower
{
    return set_flag(&termios::c_iflag, 0, IUCLC);
}
#endif

Serial&
Serial::i_through()		// read transparently
{
#if !defined(__APPLE__)
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
#if !defined(__APPLE__)
    return set_flag(&termios::c_oflag, OCRNL, OPOST|ONLCR);
#else
    return set_flag(&termios::c_oflag, 0, OPOST|ONLCR);
#endif
}

#if !defined(__APPLE__)
Serial&
Serial::o_cr2nl()		// '\n' <- '\r'
{
    return set_flag(&termios::c_oflag, ONLCR, OPOST|OCRNL);
}

Serial&
Serial::o_lower2upper()		// upper <- lower
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
    using namespace	std;

#if !defined(__APPLE__)
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
    if (::tcgetattr(_fd, &termios) == -1)
	throw runtime_error(string("TU::Serial::c_baud: tcgetattr; ")
			    + strerror(errno));
    termios.c_ispeed = termios.c_ospeed = baud;
    if (::tcsetattr(_fd, TCSANOW, &termios) == -1)
	throw runtime_error(string("TU::Serial::c_baud: tcsetattr; ")
			    + strerror(errno));
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
    using namespace	std;
    
    termios		termios;
    if (::tcgetattr(_fd, &termios) == -1)
	throw runtime_error(string("TU::Serial::set_flag: tcgetattr; ")
			    + strerror(errno));
    termios.*flag &= ~clearbits;
    termios.*flag |= setbits;
    if (::tcsetattr(_fd, TCSANOW, &termios) == -1)
	throw runtime_error(string("TU::Serial::set_flag: tcsetattr; ")
			    + strerror(errno));
    return *this;
}

}

