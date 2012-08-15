/*
 *  $Id: main.cc,v 1.3 2012-08-15 07:58:34 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include <exception>
#include "TU/TriggerGenerator.h"

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    const char*		ttyname = "/dev/ttyS0";
    extern char*	optarg;
    
    for (int c; (c = getopt(argc, argv, "d:")) != -1; )
	switch (c)
	{
	  case 'd':
	    ttyname = optarg;
	    break;
	}

    try
    {
	TriggerGenerator	trigger(ttyname);

	cerr << ">> ";
	char	command[128];
	while (cin >> command)
	{
	    switch (command[0])
	    {
	      case 'V':
		trigger.showId(cout);
		break;
	      case 'A':
		trigger.selectChannel(strtoul(command + 1, NULL, 16));
		break;
	      case 'F':
		trigger.setInterval(atoi(command + 1));
		break;
	      case 'T':
		trigger.oneShot();
		break;
	      case 'R':
		trigger.continuousShot();
		break;
	      case 'S':
		trigger.stopContinuousShot();
		break;
	      case 'I':
	      {
		  u_int	channel, interval;
		  bool	run = trigger.getStatus(channel, interval);
		  cout << "  channel: "
		       << setw(8) << setfill('0') << std::hex << channel
		       << ", interval: " << std::dec << interval
		       << ", " << (run ? "run." : "stop.") << endl;
	      }
	      break;

	      case 'h':
	      case '?':
		cout << "Commands.\n"
		     << "  V:     show ID.\n"
		     << "  A<ch>: select channels with a hex bit pattern.\n"
		     << "  F<n>:  set interval time (unit: msec).\n"
		     << "  T:     generate a single trigger pulse.\n"
		     << "  R:     generate trigger pulses continuously.\n"
		     << "  S:     stop trigger pulses.\n"
		     << "  I:     show current status.\n"
		     << endl;
		break;
		
	      default:
		trigger << command << endl;
		trigger.getline(command, sizeof(command));
		cout << command << endl;
		break;
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
