/*
 *  $Id: main.cc,v 1.2 2012-08-15 07:58:34 ueshiba Exp $
 */
#include <unistd.h>
#include <cstdlib>
#include <exception>
#include "TU/Serial.h"

namespace TU
{
/************************************************************************
*   static functions							*
************************************************************************/
static void
usage(const char* s)
{
    using namespace	std;

    cerr << "\nPerform communication test through serial lines.\n"
	 << endl;
    cerr << " Usage: " << s << " [-d <tty>] [options]\n"
	 << "\n  If the first characer of the input is \'#\', then only\n"
	 << "  the following characters are transfered to the device and\n"
	 << "  no response is expected.\n"
	 << endl;
    cerr << " General options.\n"
	 << "  -d <tty>:   specify tty device          (default: /dev/ttyS0)\n"
	 << "  -n:         translate CR to NL on input (default: do nothing)\n"
	 << "  -i:         ignore CR on input          (default: do nothing)\n"
	 << "  -N:         map NL to CR-NL on output   (default: do nothing)\n"
	 << "  -b <baud>:  set baud rate               (default: 9600)\n"
	 << "  -c <csize>: set character size          (defualt: 8)\n"
	 << "  -e:         set parity even             (default: none)\n"
	 << "  -o:         set parity odd              (default: none)\n"
	 << "  -2:         set stop bit to 2           (defualt: 1)\n"
	 << endl;
    cerr << " Other options.\n"
	 << "  -h:             print this.\n"
	 << endl;
}
    
}

/************************************************************************
*   global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;

    enum IO		{Through, NLCR, IgnCR};
    enum Parity		{None, Even , Odd};
    
    const char*		ttyname = "/dev/ttyS0";
    IO			input = Through, output = Through;
    int			baud = 9600, csize = 8;
    Parity		parity = None;
    bool		stop2 = false;
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "d:niNb:c:eo2h")) != -1; )
	switch (c)
	{
	  case 'd':
	    ttyname = optarg;
	    break;

	  // input flags.
	  case 'n':
	    input = NLCR;			// CR -> NL
	    break;
	  case 'i':
	    input = IgnCR;			// ignore CR
	    break;

	  // output flags.
	  case 'N':
	    output = NLCR;			// NL -> CR + NL
	    break;

	  // control flags.
	  case 'b':
	    baud = atoi(optarg);		// baud rate
	    break;
	  case 'c':
	    csize = atoi(optarg);		// character size
	    break;
	  case 'e':
	    parity = Even;			// even parity
	    break;
	  case 'o':
	    parity = Odd;			// odd parity
	    break;
	  case '2':
	    stop2 = true;			// stop bit 2

	  case 'h':
	    usage(argv[0]);
	    return 1;
	}

    try
    {
	Serial	serial(ttyname);

      // Set input flags.
	switch (input)
	{
	  case NLCR:
	    serial.i_cr2nl();
	    break;
	  case IgnCR:
	    serial.i_igncr();
	    break;
	  default:
	    serial.i_through();
	    break;
	}
	
      // Set output flags.
	if (output == NLCR)
	    serial.o_nl2crnl();
	else
	    serial.o_through();
	
      // Set control flags.
	serial.c_baud(baud).c_csize(csize);
	switch (parity)
	{
	  case Even:
	    serial.c_even();
	    break;
	  case Odd:
	    serial.c_odd();
	    break;
	  default:
	    serial.c_noparity();
	    break;
	}
	if (stop2)
	    serial.c_stop2();
	else
	    serial.c_stop1();
	
      // Main loop.
	cerr << ">> ";
	char	command[128];
	while (cin >> command)
	{
	    if (command[0] == '#')
		serial << (command + 1) << endl;
	    else
	    {
		serial << command << endl;
		serial.getline(command, sizeof(command));
		cout << command << endl;
	    }

	    cerr << ">> ";
	}
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
    }
	
    return 0;
}
