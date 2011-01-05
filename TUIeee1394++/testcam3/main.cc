/*
 *  $Id: main.cc,v 1.3 2011-01-05 02:05:22 ueshiba Exp $
 */
#include <cstdlib>
#include <iomanip>
#include "MyCmdWindow.h"

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    v::App			vapp(argc, argv);
    const char*			triggerDev = "/dev/ttyS0";
    Ieee1394Camera::Type	type = Ieee1394Camera::Binocular;
    bool			i1394b = false;
    u_int			delay = 1;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:t:123")) != -1; )
	switch (c)
	{
	  case 'b':
	    i1394b = true;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	  case 't':
	    triggerDev = optarg;
	    break;
	  case '1':
	    type = Ieee1394Camera::Monocular;
	    break;
	  case '2':
	    type = Ieee1394Camera::Binocular;
	    break;
	}
    extern int	optind;
    u_int64_t	uniqId = 0;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
    try
    {
#if defined(UseTrigger)
	TriggerGenerator	trigger(triggerDev);
#endif
	Ieee1394Camera		camera(type, i1394b, uniqId, delay);

	cerr << "0x" << hex << setw(16) << setfill('0')
	     << camera.globalUniqueId() << dec << endl;
	
	v::MyCmdWindow	myWin(vapp, camera, type
#if defined(UseTrigger)
			      , trigger
#endif
			     );
	vapp.run();

	camera.stopContinuousShot();

    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
