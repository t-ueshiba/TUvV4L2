/*
 *  $Id: kbhit.cc,v 1.1.1.1 2002-07-25 02:14:15 ueshiba Exp $
 */
#include <termios.h>
#include <stdio.h>
#include <unistd.h>

static termios	termios_org;

void
init_kbhit()
{
    termios	termios;
    tcgetattr(0, &termios);
    termios_org = termios;
    
    termios.c_lflag &= ~(ICANON | ECHO);
    termios.c_cc[VMIN] = 0;
    termios.c_cc[VTIME] = 1;
    termios.c_iflag &= ~ICRNL;
  //    termios.c_oflag &= ~ONLCR;
    tcsetattr(0, TCSANOW, &termios);
}

void
term_kbhit()
{
    tcsetattr(0, TCSANOW, &termios_org);
}

int
kbhit()
{
    char	c;
    if (read(0, &c, 1) > 0)
	return c;
    else
	return EOF;
}
