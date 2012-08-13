/*
 *  $Id: main.cc,v 1.4 2012-08-13 07:15:07 ueshiba Exp $
 */
#include <unistd.h>
#include <stdlib.h>
#include "MyCmdWindow.h"
#include "TU/Ieee1394++.h"
#include <iomanip>

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    v::App		vapp(argc, argv);
    const char*		triggerDev = "/dev/ttyS0";
    Ieee1394Node::Speed	speed = Ieee1394Node::SPD_400M;
    u_int		delay = 1;

  // Parse command line.
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "bd:t:")) != EOF; )
	switch (c)
	{
	  case 'b':
	    speed = Ieee1394Node::SPD_800M;
	    break;
	  case 'd':
	    delay = atoi(optarg);
	    break;
	  case 't':
	    triggerDev = optarg;
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
	Ieee1394Camera		camera(Ieee1394Camera::Monocular,
				       uniqId, speed, delay);

	v::MyCmdWindow	myWin(vapp, camera
#if defined(UseTrigger)
			      , trigger
#endif
			     );
	vapp.run();

	camera.stopContinuousShot();

	cerr << "0x" << hex << setw(16) << setfill('0')
	     << camera.globalUniqueId() << dec << ' ' << camera;
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
